"""Bounded native parity on one actual head-derived 5120x8192 fixture.

Requires explicit execution and the parent's serial external resource monitor.
No install/build actions. Global regularization/H/scale are frozen once; neither
path retunes them. This is operator parity, not model calibration-quality data.
"""

import argparse
import faulthandler
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_exl3_layer import digest, local_path
from run_exl3_head_pilot import tensor_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if not args.execute or not all(config["execution"].get(k) is True for k in
                                  ("allow_local_inference", "allow_backend_probes")):
        parser.error("requires --execute and both configured local permissions")
    binary, = args.extension_dir.glob("exllamav3_ext*.so")
    if digest(binary) != args.expected_extension_sha256:
        parser.error("extension hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    log = (args.output / "events.jsonl").open("x")
    started = time.monotonic()

    def record(stage, **values):
        line = json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-started, **values), allow_nan=False)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        import torch
        import torch.utils.cpp_extension as cpp
        from safetensors import safe_open
        from safetensors.torch import save_file
        from quantlab.methods.exl3.oracle import decode_trellis
        torch.set_num_threads(2)

        def no_build(*a, **kw):
            raise RuntimeError("C++ JIT is forbidden")
        cpp.load = cpp.load_inline = no_build
        spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
        ext = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3_ext"] = ext
        spec.loader.exec_module(ext)
        sys.path.insert(0, str(args.source_dir))
        from exllamav3.modules.quant.exl3_lib.quantize import (
            finalize_capture_H, regularize, ldlq, pack_trellis,
            block_proxy_error_streamed, ldlq_packed_columns,
        )
        device = "cuda:0"
        props = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(10 * 2**30 / props.total_memory)
        torch.cuda.reset_peak_memory_stats()
        record("runtime", device=props.name, torch=torch.__version__, hip=torch.version.hip,
               extension_sha256=digest(binary), harness_sha256=digest(Path(__file__)),
               quantizer_sha256=digest(args.source_dir / "exllamav3/modules/quant/exl3_lib/quantize.py"),
               oracle_sha256=digest(ROOT / "src/quantlab/methods/exl3/oracle.py"),
               scope="actual head slice; synthetic Hessian; native layout parity only")
        model = local_path(config["paths"]["bf16_model_dir"])
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        key = "lm_head.weight"
        shard = model / index[key]
        provenance = tensor_provenance(shard, key)
        with safe_open(shard, framework="pt", device="cpu") as file:
            original = file.get_slice(key)[:8192, :]
            assert original.dtype == torch.bfloat16 and tuple(original.shape) == (8192, 5120)
            slice_sha = hashlib.sha256(original.view(torch.int16).numpy().tobytes()).hexdigest()
            weight = original.half().T.contiguous().to(device).float()
            del original
        del file
        record("source", source_provenance=provenance, output_rows=[0, 8192], input_columns=[0, 5120],
               bf16_slice_sha256=slice_sha)
        qa = {"K": 4, "seed": 20260909, "apply_out_scales": None,
              "devices": [device], "compact_cpu_buffers": True, "memory_diagnostics": True}
        with torch.inference_mode():
            cal = torch.randn(512, 5120, generator=torch.Generator().manual_seed(100))
            h_data = {"H": (cal.T @ cal).to(device), "count": 512, "finalized": False,
                      "device": device, "first_key": key}
            del cal
            torch.manual_seed(qa["seed"])
            fallback, h, lower, su, diag = finalize_capture_H(h_data, qa, False)
            if fallback:
                raise RuntimeError("fixture Hessian fell back instead of LDLQ")
            lower = lower.to(device)
            sv = (torch.randn(8192, device=device).sign() + 1e-5).sign().float().unsqueeze(0)
            apply_scales, weight_r, scale, su, sv = regularize(weight, su, sv, qa, False, diag, None)
            frozen = {"weight_r": weight_r.cpu(), "H": h.cpu(), "L": lower.cpu(),
                      "suh": su.flatten().half().cpu(), "svh": sv.flatten().half().cpu()}
            del weight, weight_r, h, lower, su, sv, diag, h_data
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            save_file(frozen, str(args.output / "frozen.safetensors"))
            frozen_sha = digest(args.output / "frozen.safetensors")
            record("frozen", sha256=frozen_sha, shape=[5120, 8192], seed=qa["seed"],
                   calibration_seed=100, calibration_rows=512, global_scale=scale,
                   apply_out_scales=apply_scales, allocated_bytes=torch.cuda.memory_allocated(),
                   reserved_bytes=torch.cuda.memory_reserved())
            lower = frozen["L"].to(device)
            q_full, encoded_full = ldlq(frozen["weight_r"], lower, qa)
            proxy_full = block_proxy_error_streamed(frozen["weight_r"], q_full, frozen["H"], device)
            packed_full = pack_trellis(encoded_full.to(device), qa).cpu()
            # Freeze expected decoded values before dropping monolithic temporaries.
            expected_decoded = q_full.clone()
            del q_full, encoded_full
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            record("monolithic_complete", proxy=proxy_full)
            rng_cpu = torch.get_rng_state().clone()
            rng_gpu = torch.cuda.get_rng_state().clone()
            proxy_stream, packed_stream = ldlq_packed_columns(frozen["weight_r"], lower, frozen["H"], qa)
            rng_unchanged = torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_gpu, torch.cuda.get_rng_state())
            for name, packed in (("monolithic", packed_full), ("streamed", packed_stream)):
                save_file({"trellis": packed, "suh": frozen["suh"], "svh": frozen["svh"]},
                          str(args.output / (name + ".safetensors")))
            packed_mismatches = int((packed_full != packed_stream).sum().item())
            oracle = torch.from_numpy(decode_trellis(packed_stream.numpy(), 4))
            oracle_mismatches = int((oracle != expected_decoded).sum().item())
            oracle_finite = bool(torch.isfinite(oracle).all().item())
            del oracle
            decode_results = {}
            for name, packed in (("monolithic", packed_full), ("streamed", packed_stream)):
                gpu_packed = packed.to(device)
                decoded = torch.empty((5120, 8192), dtype=torch.half, device=device)
                ext.reconstruct(decoded, gpu_packed, 4, False, False)
                torch.cuda.synchronize()
                actual = decoded.cpu()
                decode_results[name] = dict(mismatches=int((actual != expected_decoded).sum().item()),
                                            finite=bool(torch.isfinite(actual).all().item()))
                del gpu_packed, decoded, actual
            relative_proxy = abs(proxy_stream-proxy_full) / max(abs(proxy_full), 1e-20)
            sizes = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
            result = dict(packed_int16_mismatches=packed_mismatches, scales_shared_from_frozen=True,
                          proxy_monolithic=proxy_full, proxy_streamed=proxy_stream,
                          proxy_relative_difference=relative_proxy, rng_unchanged=rng_unchanged,
                          independent_cpu_decode_mismatches=oracle_mismatches,
                          independent_cpu_decode_finite=oracle_finite, independently_decoded_values=5120*8192,
                          native_decode=decode_results, frozen_sha256=frozen_sha,
                          monolithic_sha256=digest(args.output / "monolithic.safetensors"),
                          streamed_sha256=digest(args.output / "streamed.safetensors"),
                          output_bytes=sizes, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            passed = (packed_mismatches == 0 and rng_unchanged and math.isfinite(proxy_full)
                      and oracle_mismatches == 0 and oracle_finite
                      and math.isfinite(proxy_stream) and relative_proxy <= 1e-5
                      and all(v["mismatches"] == 0 and v["finite"] for v in decode_results.values())
                      and sizes < 2**30)
            (args.output / "result.json").write_text(json.dumps(dict(passed=passed, **result), indent=2, allow_nan=False))
            record("compared", passed=passed, **result)
            if not passed:
                raise RuntimeError("streamed native parity failed; no retuning or fixture substitution")
    except Exception as error:
        record("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        log.close()


if __name__ == "__main__":
    main()
