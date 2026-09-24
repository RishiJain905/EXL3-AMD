"""Forward-only recovery of MiMo's saved H6 head; never encodes or commits it.

An external supervisor must enforce the existing GPU/RAM/time limits. Inputs
are an audited index-34 checkpoint and all 35 completed packed module files.
Each of the five local reference comparisons is saved separately to bound RAM.
The coordinator independently verifies and assembles the uncommitted checkpoint.
"""
import argparse
import datetime
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(value):
    value = str(value).replace("\\", "/")
    if sys.platform == "linux" and len(value) > 2 and value[1] == ":":
        return Path("/mnt") / value[0].lower() / value[3:]
    return Path(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object, parse_constant=reject_constant)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def check_job(job):
    if (not isinstance(job, dict) or set(job) != {"next_module_idx", "q_strategy", "bad_rows"}
            or type(job["next_module_idx"]) is not int or job["next_module_idx"] != 34
            or job["q_strategy"] is not None or job["bad_rows"] != []):
        raise ValueError("Recovery requires the intact index-34 checkpoint with no rejected rows")


def check_geometry(head):
    if (head.key != "lm_head" or head.in_features != 4096 or head.out_features != 248320
            or head.num_slices != 1 or head.trim_padded_out or head.out_dtype is not None
            or not head.caps.get("logits_output")):
        raise ValueError("Head differs from the audited MiMo geometry/dtype")


def check_packed_manifest(manifest):
    expected = {"lm_head.safetensors", "model.language_model.embed_tokens.safetensors",
                "model.language_model.norm.safetensors"}
    expected |= {f"model.language_model.layers.{i}.safetensors" for i in range(32)}
    if not isinstance(manifest, list) or len(manifest) != 35:
        raise ValueError("Exactly 35 packed module records required")
    found = set()
    for row in manifest:
        if not isinstance(row, dict) or set(row) != {"file", "bytes", "sha256"}:
            raise ValueError("Malformed packed module record")
        name, size, checksum = row["file"], row["bytes"], row["sha256"]
        if not isinstance(name, str) or name not in expected or name in found:
            raise ValueError("Unexpected or duplicate packed module filename")
        if type(size) is not int or size <= 0:
            raise ValueError("Invalid packed module length")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid packed module checksum")
        found.add(name)
    if found != expected:
        raise ValueError("Packed module set differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "recover"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    local = tomllib.loads(args.config.read_text(encoding="utf-8"))
    if not args.execute or not all(local["execution"].get(k) is True for k in
                                  ("allow_local_inference", "allow_backend_probes")):
        parser.error("requires --execute and both local permissions")
    source = local_path(local["paths"]["bf16_model_dir"]).resolve()
    work, recovery, output = (p.resolve() for p in (args.work, args.recovery, args.output))
    for input_path in (source, work, recovery):
        if output == input_path or input_path in output.parents or output in input_path.parents:
            parser.error("output must be separate and nonnested with inputs")
    if output.exists() or args.output.is_symlink():
        parser.error("new output directory required")
    if (work / "ckpt_new").exists():
        parser.error("pending checkpoint exists; coordinator inspection required")
    check_job(read_json(work / "ckpt/job.json"))
    evidence = read_json(recovery / "verification.json")
    if (evidence.get("failed_attempt") != "b50-convert-34-34-memory-recovery"
            or evidence.get("previous_34_packed_files_unchanged") is not True
            or evidence.get("committed_checkpoint_unchanged") is not True
            or evidence.get("packed_file_count") != 35
            or evidence.get("packed_groups") != 201
            or evidence.get("failed_windows", {}).get("reason") != "windows_memory_guard"
            or evidence.get("failed_monitor", {}).get("exit_code") != -15
            or evidence.get("conversion_complete") is not False):
        parser.error("audited post-save RAM-stop evidence required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    events = (output / "events.jsonl").open("x", encoding="utf-8")

    def record(stage, **values):
        entry = dict(stage=stage, elapsed_seconds=time.monotonic()-started, **values)
        line = json.dumps(entry, allow_nan=False)
        events.write(line + "\n"); events.flush()
        print(line, flush=True)

    try:
        (output / "harness.py").write_bytes(Path(__file__).read_bytes())
        identity = read_json(work / "conversion_identity.json")
        for name, key in (("conversion_identity.json", "identity_sha256"), ("args.json", "args_sha256")):
            if digest(work / name) != evidence[key]:
                raise ValueError(f"{name} changed since the audit")
        stored = read_json(work / "args.json")
        if (local_path(stored["in_dir"]).resolve() != source or stored["bits"] != 5.0
                or stored["head_bits"] != 6 or stored["codebook"] != "mul1"
                or stored["cal_rows"] != 128 or stored["cal_cols"] != 2048
                or stored["devices"] != "0" or stored["hq"] is not False):
            raise ValueError("Frozen conversion settings differ")
        for name, checksum in evidence["checkpoint_sha256"].items():
            if name not in ("job.json", "state.safetensors", "original_input_ids.safetensors"):
                raise ValueError("Unexpected checkpoint filename")
            if digest(work / "ckpt" / name) != checksum:
                raise ValueError(f"Checkpoint {name} changed")
        if set(evidence["checkpoint_sha256"]) != {"job.json", "state.safetensors", "original_input_ids.safetensors"}:
            raise ValueError("Incomplete checkpoint hashes")
        manifest_path = recovery / "qtensors-before-postsave-recovery.json"
        if digest(manifest_path) != evidence["packed_manifest_sha256"]:
            raise ValueError("Packed manifest changed")
        manifest = read_json(manifest_path); check_packed_manifest(manifest)
        for row in manifest:
            path = work / "qtensors" / row["file"]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"Packed module changed: {row['file']}")
        from run_exl3_conversion import source_identity
        if source_identity(source) != identity["source_files"]:
            raise ValueError("BF16 source/metadata hashes differ")
        vendor = ROOT / "vendor/rocm-exl3"
        code = {str(p.relative_to(vendor)): digest(p) for p in sorted((vendor / "exllamav3").rglob("*.py"))}
        checks = dict(code_sha256=code, wrapper_sha256=digest(ROOT / "scripts/run_exl3_conversion.py"),
            source_adapter_sha256=digest(ROOT / "scripts/inspect_exl3_source.py"),
            compatibility_sha256=digest(ROOT / "src/quantlab/methods/exl3/compat.py"),
            calibration_sha256=digest(local_path(stored["cal_data"])))
        if any(identity.get(key) != value for key, value in checks.items()):
            raise ValueError("Frozen code/calibration identity differs")
        if identity.get("compact_cpu_buffers") is not True or identity.get("noncoop_compat") is not True:
            raise ValueError("Unexpected original conversion dispatch")
        extension_dir = local_path(local["runtime"]["extension_dir"])
        binary, = extension_dir.glob("exllamav3_ext*.so")
        if digest(binary) != identity["extension_sha256"] or digest(binary) != local["runtime"]["extension_sha256"]:
            raise ValueError("Native extension hash differs")
        record("identity_verified", packed_files=35, quantization_identity_sha256=evidence["identity_sha256"],
               recovery_worker_sha256=digest(Path(__file__)))
        os.environ["EXL3_BC_ATTN"] = "0"; os.environ["EXL3_GEMV"] = "2"
        faulthandler.enable()
        import torch
        import torch.utils.cpp_extension as cpp
        def no_build(*unused, **kwargs):
            raise RuntimeError("C++ JIT build forbidden")
        cpp.load = cpp.load_inline = no_build
        torch.set_num_threads(2)
        spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
        extension = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3_ext"] = extension; spec.loader.exec_module(extension)
        sys.path.insert(0, str(vendor))
        from exllamav3 import Model
        from exllamav3.conversion.convert_model import get_state_error
        from exllamav3.loader.safetensors_alt import safe_open, save_file
        from exllamav3.modules.quant import LinearFP16, LinearEXL3
        from exllamav3.util.memory import malloc_trim
        from inspect_exl3_source import mapped_text_config
        from quantlab.methods.exl3.compat import install, prepare_loaded_module
        torch.cuda.set_per_process_memory_fraction(10 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        configs, heads = [], []
        with torch.inference_mode():
            for unused in range(2):
                config, _ = mapped_text_config(source); install(config)
                model = Model.from_config(config)
                if len(model.modules) != 35 or model.calibration_all_experts:
                    raise ValueError("Unexpected model/calibration layout")
                head = model.modules[34]; check_geometry(head)
                configs.append(config); heads.append(head)
            base, packed = heads
            deferred = base.can_defer_load()
            if deferred: configs[0].stc.begin_deferred_load()
            try:
                base.load(torch.device("cuda:0"))
            finally:
                if deferred: configs[0].stc.end_deferred_load()
            prepare_loaded_module(base); configs[0].stc.close()
            with safe_open(str(work / "qtensors/lm_head.safetensors"), framework="pt", device="cpu") as handle:
                tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            if (set(tensors) != {"lm_head.suh", "lm_head.svh", "lm_head.trellis", "lm_head.mul1"}
                    or tensors["lm_head.mul1"].view(torch.uint32).item() != 0x83DCD12D
                    or any(not torch.isfinite(t).all().item() for t in tensors.values() if t.is_floating_point())):
                raise ValueError("Packed head codebook/scales differ")
            configs[1].stc.set_new_tensors(tensors)
            try:
                packed.load(torch.device("cuda:0"), source=tensors)
            finally:
                configs[1].stc.set_new_tensors(None)
            prepare_loaded_module(packed); configs[1].stc.close(); del tensors
            if not isinstance(base.inner, LinearFP16) or not isinstance(packed.inner, LinearEXL3) or packed.inner.K != 6:
                raise ValueError("Reference/packed loader selected the wrong storage type")
            record("heads_loaded", reference="BF16-source loaded as FP16", packed="saved mul1 H6",
                   geometry=[4096,248320], device=torch.cuda.get_device_properties(0).name)
            rows = []; sums = [0.0, 0.0, 0.0]
            with safe_open(str(work / "ckpt/state.safetensors"), framework="pt", device="cpu") as states, \
                 safe_open(str(work / "ckpt/original_input_ids.safetensors"), framework="pt", device="cpu") as ids:
                expected = {f"tensor.{i}" for i in range(128)}
                if set(states.keys()) != expected or set(ids.keys()) != expected:
                    raise ValueError("Checkpoint row set differs")
                for i in range(1 if args.mode == "probe" else 5):
                    hidden = states.get_tensor(f"tensor.{i}"); token_ids = ids.get_tensor(f"tensor.{i}")
                    if (hidden.dtype != torch.float16 or tuple(hidden.shape) != (1,2048,4096)
                            or token_ids.dtype != torch.int64 or tuple(token_ids.shape) != (1,2048)
                            or not hidden.is_contiguous() or not torch.isfinite(hidden).all().item()
                            or token_ids.min().item() < 0 or token_ids.max().item() >= 248077):
                        raise ValueError("Invalid committed row")
                    if args.mode == "probe": hidden = hidden[:,:8].contiguous(); token_ids = token_ids[:,:8].contiguous()
                    params = dict(attn_mode="flash_attn_nc", input_ids=token_ids, quant_preserve={})
                    x = base.prepare_for_device(hidden, params)
                    ref_gpu = base.forward(x, params)
                    if args.mode == "probe":
                        captured = base.forward(x, dict(params, capture={}, activate_all_experts=False))
                        if not torch.equal(ref_gpu, captured): raise ValueError("Capture changed reference output")
                        del captured
                    reference = ref_gpu.cpu(); del ref_gpu, x
                    params = dict(attn_mode="flash_attn_nc", input_ids=token_ids, quant_preserve={})
                    x = packed.prepare_for_device(hidden, params)
                    result_gpu = packed.forward(x, params)
                    if args.mode == "probe":
                        repeated = packed.forward(x, params)
                        if not torch.equal(result_gpu, repeated): raise ValueError("Packed forward is not repeatable")
                        del repeated
                    result = result_gpu.cpu(); del result_gpu, x
                    expected_shape = (1,8 if args.mode == "probe" else 2048,248320)
                    if (reference.dtype != torch.float16 or result.dtype != torch.float16
                            or tuple(reference.shape) != expected_shape or tuple(result.shape) != expected_shape
                            or not torch.isfinite(reference).all().item() or not torch.isfinite(result).all().item()):
                        raise ValueError("Nonfinite or unexpected head output; no row may be excluded")
                    errors = get_state_error(result, reference)
                    if len(errors) != 3 or not all(math.isfinite(v) for v in errors):
                        raise ValueError("Nonfinite head diagnostics")
                    for j, value in enumerate(errors): sums[j] += value
                    row = dict(row=i, dtype="F16", shape=list(result.shape),
                               rfn=errors[0], cosine_error=errors[1], sqnr_db=errors[2])
                    del reference, hidden, token_ids, params
                    if args.mode == "recover":
                        path = output / f"row-{i:03}.safetensors"
                        if path.exists(): raise FileExistsError(path)
                        save_file({"tensor":result}, str(path))
                        row.update(file=path.name, sha256=digest(path), bytes=path.stat().st_size)
                    del result; malloc_trim()
                    rows.append(row); record("row_complete", **row)
            base.unload(); packed.unload(); torch.cuda.synchronize(); malloc_trim()
        if digest(work / "conversion_identity.json") != evidence["identity_sha256"]:
            raise ValueError("Conversion identity changed during recovery")
        if digest(work / "qtensors/lm_head.safetensors") != evidence["head"]["sha256"]:
            raise ValueError("Packed head changed during recovery")
        check_job(read_json(work / "ckpt/job.json"))
        result = dict(schema="mimo-head-postsave-v1", mode=args.mode, complete=True,
            recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
            checkpoint_sha256=evidence["checkpoint_sha256"],
            quantization_identity_sha256=evidence["identity_sha256"], packed_head_sha256=evidence["head"]["sha256"],
            recovery_worker_sha256=digest(Path(__file__)), packed_manifest_sha256=evidence["packed_manifest_sha256"],
            mean_rfn=sums[0]/len(rows), mean_cosine_error=sums[1]/len(rows), mean_sqnr_db=sums[2]/len(rows),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started,
            capture_output_exact_match=True if args.mode == "probe" else None,
            repeated_packed_output_exact_match=True if args.mode == "probe" else None,
            checkpoint_committed=False, packed_files_written=False, rows_excluded=0)
        write_new(output / ("probe-completed.json" if args.mode == "probe" else "rows-completed.json"), result)
        record("finished", mode=args.mode, rows=len(rows), checkpoint_committed=False)
    finally:
        events.close()


if __name__ == "__main__":
    main()
