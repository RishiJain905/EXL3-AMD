"""Fresh-process EXL3 candidate smoke. External runner enforces time/headroom.

Uses a fixed raw completion prompt, FP16 KV cache, and experimental noncooperative
dispatch. Allocator statistics are not total dedicated/shared GPU measurements.
"""

import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import tomllib
import traceback

ROOT = Path(__file__).resolve().parents[1]
HARNESS_SOURCE = Path(__file__).read_bytes()
sys.path.insert(0, str(ROOT / "src"))
from inspect_exl3_source import mapped_text_config

PROMPT = "The capital of France is"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_logits(logits, token_count, vocab_size, actual_vocab_size=None):
    """Validate raw logits, or returned logits with the exact tokenizer tail mask."""
    import torch

    if logits is None or tuple(logits.shape) != (1, token_count, vocab_size):
        raise ValueError("Missing or unexpected streamed logits shape")
    if token_count < 1 or (actual_vocab_size is not None and not 0 < actual_vocab_size <= vocab_size):
        raise ValueError("Expected at least one row and a nonempty valid vocabulary")
    checked = {"checked_logit_count": logits.numel(), "checked_logit_rows": token_count,
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "nan_count": int(torch.isnan(logits).sum().item()),
            "positive_infinity_count": int(torch.isposinf(logits).sum().item()),
            "negative_infinity_count": int(torch.isneginf(logits).sum().item())}
    if actual_vocab_size is not None:
        valid = logits[..., :actual_vocab_size]
        tail = logits[..., actual_vocab_size:]
        checked.update(valid_vocab_finite=bool(torch.isfinite(valid).all().item()),
                       expected_masked_tail_count=tail.numel(),
                       masked_tail_count=int(torch.isneginf(tail).sum().item()))
        checked["logits_valid"] = (checked["valid_vocab_finite"] and
                                    checked["masked_tail_count"] == tail.numel() and
                                    checked["nan_count"] == checked["positive_infinity_count"] == 0)
    else:
        checked["logits_valid"] = checked["logits_finite"]
    return checked


def observe_sampler_input(sampler, callback):
    """Observe before CustomSampler.forward mutates the input tail in-place."""
    original = sampler.forward
    def forward(logits, *args, **kwargs):
        callback(logits)
        return original(logits, *args, **kwargs)
    sampler.forward = forward


def observe_mtp_head(draft, model, callback):
    """Observe Qwen MTP's shared head before its direct argmax, without sampling changes."""
    original_sample = draft.sample_from_state
    def sample_from_state(*args, **kwargs):
        head = model.modules[model.logit_layer_idx]
        original_forward = head.forward
        def forward(*head_args, **head_kwargs):
            logits = original_forward(*head_args, **head_kwargs)
            callback(logits)
            return logits
        head.forward = forward
        try:
            return original_sample(*args, **kwargs)
        finally:
            head.forward = original_forward
    draft.sample_from_state = sample_from_state


def candidate_preflight(candidate):
    """Cheap checks before runtime import or expensive candidate hashing."""
    for name in ("config.json", "quantization_config.json", "tokenizer.json"):
        if not (candidate / name).is_file():
            raise ValueError(f"Incomplete candidate: missing {name}")
    config = json.loads((candidate / "config.json").read_text())
    if config.get("quantization_config", {}).get("quant_method") != "exl3":
        raise ValueError("Candidate config must identify quant_method=exl3")
    if not list(candidate.glob("*.safetensors")):
        raise ValueError("Incomplete candidate: no safetensors shards")


def check_required_tensors(config, models):
    from exllamav3.modules import Linear, Embedding, RMSNorm, GatedDeltaNet

    missing = set()
    def require(key):
        if not key or not config.stc.has_tensor(key):
            missing.add(str(key))
    for model in models:
        for module in model:
            if isinstance(module, Linear):
                if module.qmap is not None:
                    for suffix in ("trellis", "suh", "svh"):
                        require(module.key + "." + suffix)
                else:
                    require(module.key + ".weight")
            elif isinstance(module, Embedding):
                require(module.key + ".weight")
            elif isinstance(module, RMSNorm) and not module.unweighted:
                require(module.tensor_key)
            elif isinstance(module, GatedDeltaNet):
                for attr in ("key_a_log", "key_dt_bias", "key_conv1d_weight"):
                    require(getattr(module, attr))
                require(module.norm.key + ".weight")
    if missing:
        raise ValueError("Incomplete packed text/MTP candidate; missing: " + ", ".join(sorted(missing)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(8, 32), default=8)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--gpu-budget-gib", type=float, default=12)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gpu_budget_gib <= 14:
        parser.error("--gpu-budget-gib must be positive and at most 14")
    config_file = tomllib.loads(args.config.read_text())
    if not args.execute or not all(config_file["execution"].get(k) is True for k in
                                   ("allow_local_inference", "allow_backend_probes")):
        parser.error("requires --execute and both configured local permissions")
    candidate = args.candidate.resolve()
    output = args.output.resolve()
    if output == candidate or candidate in output.parents or output in candidate.parents:
        parser.error("output and candidate must be separate nonnested paths")
    candidate_preflight(candidate)
    output.mkdir(parents=True, exist_ok=False)
    (output / "harness.py").write_bytes(HARNESS_SOURCE)
    started = time.monotonic()
    torch = None
    status = {"status": "running", "mtp": args.mtp, "requested_tokens": args.tokens,
              "prompt": PROMPT, "cache_tokens": 512, "gpu_budget_gib": args.gpu_budget_gib,
              "checked_logit_count": 0, "checked_logit_rows": 0, "logits_finite": None,
              "prefill_seconds": None, "decode_seconds": None,
              "dedicated_gpu_bytes": None, "shared_gpu_bytes": None}
    events = (output / "events.jsonl").open("x", encoding="utf-8")

    def memory():
        if torch is None:
            return None
        try:
            return {"allocated_bytes": torch.cuda.memory_allocated(0),
                    "reserved_bytes": torch.cuda.memory_reserved(0),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        except Exception:
            return None

    def record(stage, **fields):
        event = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **fields}
        events.write(json.dumps(event, allow_nan=False) + "\n")
        events.flush()
        print(json.dumps(event, allow_nan=False), flush=True)

    try:
        files = [{"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(candidate.iterdir()) if p.is_file()]
        (output / "candidate-files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
        status["candidate_weight_file_bytes"] = sum(p["bytes"] for p in files if p["file"].endswith(".safetensors"))
        binary, = args.extension_dir.glob("exllamav3_ext*.so")
        if digest(binary) != args.expected_extension_sha256:
            raise ValueError("Extension differs from explicitly pinned SHA256")
        record("provenance", harness_sha256=hashlib.sha256(HARNESS_SOURCE).hexdigest(),
               extension_sha256=args.expected_extension_sha256,
               mapping_sha256=digest(ROOT / "scripts/inspect_exl3_source.py"),
               compat_sha256=digest(ROOT / "src/quantlab/methods/exl3/compat.py"),
               candidate_weight_file_bytes=status["candidate_weight_file_bytes"])
        os.environ["EXL3_BC_ATTN"] = "0"
        os.environ["EXL3_GEMV"] = "2"
        import torch as torch_module
        torch = torch_module
        import torch.utils.cpp_extension as cpp
        def no_build(*unused_args, **unused_kwargs):
            raise RuntimeError("C++ JIT build forbidden")
        cpp.load = cpp.load_inline = no_build
        torch.set_num_threads(2)
        # Match the evaluation harness: repeated asynchronous stack dumps
        # crashed this Python/Triton combination in an earlier runtime run.
        # The external supervisor enforces the timeout and resource limits.
        faulthandler.enable()
        spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
        extension = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3_ext"] = extension
        spec.loader.exec_module(extension)
        sys.path.insert(0, str(args.source_dir))
        from exllamav3 import Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
        from quantlab.methods.exl3.compat import install, prepare_loaded_module
        props = torch.cuda.get_device_properties(0)
        if args.gpu_budget_gib * 2**30 > props.total_memory:
            raise ValueError("Allocator budget exceeds device capacity")
        torch.cuda.set_per_process_memory_fraction(args.gpu_budget_gib * 2**30 / props.total_memory)
        torch.cuda.reset_peak_memory_stats(0)
        record("runtime", device=props.name, torch=torch.__version__, hip=torch.version.hip)
        cfg, _ = mapped_text_config(candidate)
        install(cfg)
        model = Model.from_config(cfg)
        if args.mtp and "mtp" not in cfg.model_classes:
            raise ValueError("MTP requested but candidate has no MTP weights")
        draft = Model.from_config(cfg, component="mtp") if "mtp" in cfg.model_classes else None
        # Verify MTP completeness even in target-only mode: a partial target
        # export must not accidentally count as the requested complete artifact.
        check_required_tensors(cfg, (model, draft) if draft is not None else (model,))
        record("candidate_checked", text_modules=len(model.modules), mtp_modules=len(draft.modules) if draft else 0)
        cache = Cache(model, max_num_tokens=512, max_batch_size=1, max_history=2 if args.mtp else 0)
        draft_cache = Cache(draft, max_num_tokens=512, max_batch_size=1) if args.mtp else None
        torch.cuda.synchronize()
        load_started = time.monotonic()
        if args.mtp:
            draft.load(device="cuda:0")
            for module in draft.modules:
                prepare_loaded_module(module)
        model.load(device="cuda:0")
        for module in model.modules:
            prepare_loaded_module(module)
        torch.cuda.synchronize()
        status["load_seconds"] = time.monotonic() - load_started
        status["embedding_device"] = str(model.modules[0].embedding.weight.device)
        record("loaded", load_seconds=status["load_seconds"], embedding_device=status["embedding_device"],
               allocator=memory(), text_module_devices=[str(m.device) for m in model.modules])
        tokenizer = Tokenizer.from_config(cfg)
        status["actual_vocab_size"] = tokenizer.actual_vocab_size
        ids = tokenizer.encode(PROMPT, add_bos=False, add_eos=False, encode_special_tokens=False)
        if ids.numel() + args.tokens + 2 > 512:
            raise ValueError("Prompt and generation exceed the smoke cache")
        status["input_token_ids"] = ids.flatten().tolist()
        generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=256,
                              max_q_size=3, draft_model=draft if args.mtp else None,
                              draft_cache=draft_cache, num_draft_tokens=2 if args.mtp else None,
                              enable_defrag=False, cpu_cache_size=0, recurrent_cache_size=1024**3,
                              record_draft_stats=True, dynamic_draft_tokens=False)
        # Pinned Job subtracts one in __init__ and subtracts the draft width
        # again at its receive_sample stopping condition (job.py:212,822).
        # Compensate that API boundary to request exactly the CLI token count.
        job_token_limit = args.tokens + 1 + generator.num_draft_tokens
        status["job_max_new_tokens_argument"] = job_token_limit
        status["padded_vocab_size"] = generator.padded_vocab_size
        status["raw_logit_checks"] = {"target": 0, "draft": 0}
        status["raw_logit_rows"] = {"target": 0, "draft": 0}
        status["returned_masked_tail_count"] = 0
        record("logit_boundaries", actual_vocab_size=tokenizer.actual_vocab_size,
               padded_vocab_size=generator.padded_vocab_size,
               expected_masked_tail_per_row=generator.padded_vocab_size - tokenizer.actual_vocab_size)
        def check_raw(logits, component):
            checked = inspect_logits(logits, logits.shape[1], generator.padded_vocab_size)
            # All reductions complete via .item() before the sampler can mutate
            # this tensor; only scalars are retained, so no clone is necessary.
            status["raw_logit_checks"][component] += 1
            status["raw_logit_rows"][component] += checked["checked_logit_rows"]
            record("raw_logits", component=component, **checked)
            if not checked["logits_valid"]:
                status["nonfinite_logit_diagnostic"] = {"component": component, **checked}
                raise RuntimeError(f"Nonfinite raw {component} logits before sampling")
        sampler = ArgmaxSampler()
        observe_sampler_input(sampler, lambda logits: check_raw(logits, "target"))
        if args.mtp:
            observe_mtp_head(draft, model, lambda logits: check_raw(logits, "draft"))
        job = Job(input_ids=ids, max_new_tokens=job_token_limit, sampler=sampler,
                  stop_conditions=[], token_healing=False, return_logits=True)
        token_ids = []
        torch.cuda.synchronize()
        generation_started = time.monotonic()
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                emitted = result.get("token_ids")
                emitted = emitted.flatten().tolist() if emitted is not None else []
                if emitted:
                    checked = inspect_logits(result.get("logits"), len(emitted), generator.padded_vocab_size,
                                             tokenizer.actual_vocab_size)
                    status["checked_logit_count"] += checked["checked_logit_count"]
                    status["checked_logit_rows"] += checked["checked_logit_rows"]
                    status["logits_finite"] = checked["logits_finite"]
                    status["logits_valid"] = checked["logits_valid"]
                    status["returned_masked_tail_count"] += checked["masked_tail_count"]
                    record("returned_logits", token_ids=emitted, **checked)
                    if not checked["logits_valid"]:
                        status["nonfinite_logit_diagnostic"] = checked
                        record("nonfinite_logits", token_ids=emitted, **checked)
                        raise RuntimeError("Invalid generated-token logits beyond expected tokenizer tail mask")
                token_ids.extend(emitted)
                record("generation", token_ids=emitted, text=result.get("text", ""),
                       eos=result.get("eos"), eos_reason=result.get("eos_reason"),
                       checked_logit_count=status["checked_logit_count"], logits_finite=status["logits_finite"],
                       logits_valid=status.get("logits_valid"),
                       accepted_draft_tokens=result.get("accepted_draft_tokens"),
                       rejected_draft_tokens=result.get("rejected_draft_tokens"))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - generation_started
        status.update({"status": "completed", "output_token_ids": token_ids,
                       "output_text": job.full_completion, "emitted_token_count": len(token_ids),
                       "generation_end_to_end_seconds": elapsed,
                       "end_to_end_tokens_per_second": len(token_ids) / elapsed,
                       "draft_stats": job.draft_stats,
                       "accepted_draft_tokens": getattr(job, "accepted_draft_tokens", None) if args.mtp else None,
                       "rejected_draft_tokens": getattr(job, "rejected_draft_tokens", None) if args.mtp else None})
        if len(token_ids) != args.tokens:
            raise RuntimeError("Emitted token count differs from requested fixed-length smoke")
        if not status["raw_logit_checks"]["target"] or (args.mtp and not status["raw_logit_checks"]["draft"]):
            raise RuntimeError("Expected raw logit boundary was not observed")
        record("completed", **status)
    except BaseException as error:
        status.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        record("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        status.update({"process_wall_seconds": time.monotonic() - started, "allocator": memory(),
                       "timing_scope": "Synchronized end-to-end generation including prefill; no prefill/decode segmentation. Process wall begins after output creation.",
                       "dispatch_scope": "Experimental native single-row GEMV, rowwise small batches, and per-layer dequantization plus GEMM. CPU embedding preference retained."})
        (output / "result.json").write_text(json.dumps(status, indent=2, allow_nan=False), encoding="utf-8")
        events.close()


if __name__ == "__main__":
    main()
