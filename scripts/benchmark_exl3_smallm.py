"""Compare retained packed EXL3 layers under native small-M dispatch.

Operator diagnostics only: native full batch (EXL3_SMALLM=1) versus rowwise
ordinary GEMV and versus reconstruct=True packed reference. No BF16-fidelity
claim. External runner enforces time/headroom; this script caps Torch to 10GiB.
Requires the pinned experimental binary; never installs or builds.
"""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import tomllib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
HARNESS_SOURCE = Path(__file__).read_bytes()

DEFAULT_BATCHES_TEXT = "1,2,3"
ALLOWED_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
WARMUPS = 3
TIMING_GROUPS = 5
DEFAULT_TIMING_REPEATS = 20
MAX_TIMING_REPEATS = 100
ROWWISE_REL_GATE = 1e-3
RECONSTRUCT_REL_GATE = 5e-3
GPU_BUDGET_BYTES = 10 * 2**30
FAMILIES = ("zeros", "gaussian", "alternating")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def layer_artifact(directory):
    paths = [Path(directory) / name for name in ("layer.safetensors", "head.safetensors")]
    present = [path for path in paths if path.is_file()]
    if not present:
        raise ValueError(f"missing layer.safetensors or head.safetensors in {directory}")
    if len(present) != 1:
        raise ValueError(f"ambiguous packed artifact in {directory}")
    return present[0]


def parse_batches_text(value):
    """Parse --batches subset of 1,2,3,4,5,6,7,8,9 into sorted unique ints."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("--batches must be a non-empty subset of 1,2,3,4,5,6,7,8,9")
    text = value if isinstance(value, str) else str(value)
    parts = [p.strip() for p in text.split(",")]
    if not parts or any(not p for p in parts):
        raise ValueError("--batches must be a non-empty subset of 1,2,3,4,5,6,7,8,9")
    try:
        batches = [int(p) for p in parts]
    except ValueError:
        raise ValueError("--batches must be a non-empty subset of 1,2,3,4,5,6,7,8,9")
    if any(b not in ALLOWED_BATCHES for b in batches):
        raise ValueError("--batches must be a non-empty subset of 1,2,3,4,5,6,7,8,9")
    if len(set(batches)) != len(batches):
        raise ValueError("--batches must be a non-empty subset of 1,2,3,4,5,6,7,8,9 without repeats")
    return sorted(batches)


def parse_timing_repeats_value(value):
    try:
        repeats = int(value)
    except (TypeError, ValueError):
        raise ValueError("--timing-repeats must be between 1 and 100")
    if not 1 <= repeats <= MAX_TIMING_REPEATS:
        raise ValueError("--timing-repeats must be between 1 and 100")
    return repeats


def execution_allowed(config, execute):
    if not execute:
        return False
    try:
        execution = config["execution"]
    except (KeyError, TypeError):
        return False
    if not isinstance(execution, dict):
        return False
    return all(execution.get(k) is True for k in ("allow_local_inference", "allow_backend_probes"))


def check_output_exclusive(output, inputs):
    resolved_output = Path(output).resolve()
    for candidate in inputs:
        resolved_input = Path(candidate).resolve()
        if resolved_output == resolved_input:
            raise ValueError("output and inputs must be separate nonnested paths")
        if resolved_input in resolved_output.parents or resolved_output in resolved_input.parents:
            raise ValueError("output and inputs must be separate nonnested paths")


def validate_encoding_meta(meta):
    if not isinstance(meta, dict):
        raise ValueError("encoding.json must be a JSON object")
    bits = meta.get("bits")
    if bits not in (2, 3, 4):
        raise ValueError("encoding.json bits must be 2, 3 or 4")
    sha = meta.get("artifact_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("encoding.json artifact_sha256 must be a 64-char hex digest")
    try:
        bytes.fromhex(sha)
    except ValueError:
        raise ValueError("encoding.json artifact_sha256 must be a 64-char hex digest")
    key = meta.get("key")
    if key is not None and not isinstance(key, str):
        raise ValueError("encoding.json key must be a string when present")
    return key, bits, sha.lower()


def infer_dimensions(suh_len, svh_len, trellis_shape, bits):
    """Infer (in_features, out_features) from packed scale lengths."""
    if bits not in (2, 3, 4):
        raise ValueError("bits must be 2, 3 or 4")
    for name, length in (("suh", suh_len), ("svh", svh_len)):
        if not isinstance(length, int) or length <= 0 or length % 128:
            raise ValueError(f"{name} length must be a positive multiple of 128")
    shape = tuple(int(v) for v in trellis_shape)
    expected = (suh_len // 16, svh_len // 16, 16 * bits)
    if shape != expected:
        raise ValueError(f"trellis shape {shape} mismatches suh/svh/bits {expected}")
    return suh_len, svh_len


def compare_pair(actual, ref):
    """Compare two CPU tensors; never emits NaN/Infinity.

    Returns dict with shape_match, finite flags, relative_l2, max_abs,
    exact_match. Zero reference yields 0.0 when error is also zero, else null.
    Undefined comparisons return null and fail the numerical gate.
    """
    try:
        actual_shape = tuple(actual.shape)
        ref_shape = tuple(ref.shape)
    except Exception:
        return {"shape_match": False, "actual_finite": False, "ref_finite": False,
                "both_finite": False, "relative_l2": None, "max_abs": None, "exact_match": False}
    shape_match = actual_shape == ref_shape
    try:
        actual_finite = bool(actual.isfinite().all().item())
        ref_finite = bool(ref.isfinite().all().item())
    except Exception:
        actual_finite = ref_finite = False
    both_finite = actual_finite and ref_finite
    if not shape_match or not both_finite:
        return {"shape_match": bool(shape_match), "actual_finite": bool(actual_finite),
                "ref_finite": bool(ref_finite), "both_finite": bool(both_finite),
                "relative_l2": None, "max_abs": None, "exact_match": False}
    try:
        exact_match = bool((actual.float() == ref.float()).all().item())
        diff = actual.float() - ref.float()
        err_sum = float(diff.square().sum().item())
        ref_sum = float(ref.float().square().sum().item())
        max_abs = float(diff.abs().max().item())
    except Exception:
        return {"shape_match": True, "actual_finite": True, "ref_finite": True,
                "both_finite": True, "relative_l2": None, "max_abs": None, "exact_match": False}
    if not (math.isfinite(err_sum) and math.isfinite(ref_sum) and math.isfinite(max_abs)):
        return {"shape_match": True, "actual_finite": False, "ref_finite": False,
                "both_finite": False, "relative_l2": None, "max_abs": None, "exact_match": False}
    if ref_sum == 0.0:
        relative_l2 = 0.0 if err_sum == 0.0 else None
    else:
        relative_l2 = math.sqrt(err_sum / ref_sum) if err_sum else 0.0
        if not math.isfinite(relative_l2):
            relative_l2 = None
    return {"shape_match": True, "actual_finite": True, "ref_finite": True,
            "both_finite": True, "relative_l2": float(relative_l2) if relative_l2 is not None else None,
            "max_abs": float(max_abs), "exact_match": bool(exact_match)}


def median_values(values):
    if not values:
        raise ValueError("median requires at least one sample")
    result = float(statistics.median(values))
    if not math.isfinite(result):
        raise ValueError("median is non-finite")
    return result


def mark_failed(status, error):
    status["status"] = "failed"
    status["error_type"] = type(error).__name__
    status["error"] = str(error)
    return status


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--input", dest="inputs", type=Path, action="append", required=True,
                   help="Retained layer directory with layer.safetensors and encoding.json (repeatable)")
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--extension-dir", type=Path, required=True)
    p.add_argument("--expected-extension-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", default=DEFAULT_BATCHES_TEXT,
                   help="Comma-separated subset of 1,2,3,4,5,6,7,8,9 (default 1,2,3)")
    p.add_argument("--timing-repeats", type=int, default=DEFAULT_TIMING_REPEATS,
                   help="Timed repetitions per method per group, 1..100 (default 20)")
    p.add_argument("--correctness-only", action="store_true",
                   help="Skip Gaussian operator timing; correctness comparisons only")
    p.add_argument("--fp32-output", action="store_true",
                   help="Exercise native FP32 output, as used by the model head")
    p.add_argument('--smallm-kernel', choices=('dot','wmma','wmma-register'), default='dot')
    p.add_argument('--head-warps', type=int, choices=(1,4,8,16))
    p.add_argument("--execute", action="store_true")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        batches = parse_batches_text(args.batches)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        timing_repeats = parse_timing_repeats_value(args.timing_repeats)
    except ValueError as exc:
        parser.error(str(exc))
    config = tomllib.loads(args.config.read_text())
    if not execution_allowed(config, args.execute):
        parser.error("requires --execute and both configured local permissions")
    try:
        check_output_exclusive(args.output, args.inputs)
    except ValueError as exc:
        parser.error(str(exc))
    for layer_dir in args.inputs:
        layer_artifact(layer_dir)
        if not (Path(layer_dir) / "encoding.json").is_file():
            raise ValueError(f"missing encoding.json in {layer_dir}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(HARNESS_SOURCE)
    # New native contract only: SMALLM routes eligible m=2/3/4/5 through the
    # noncooperative kernel; GEMV=2 preserves the ordinary m=1 baseline.
    # Do NOT install quantlab.compat's row-splitting wrapper here.
    os.environ["EXL3_SMALLM"] = "1"
    os.environ["EXL3_GEMV"] = "2"
    started = time.monotonic()
    status = {"status": "running", "batches": batches, "timing_repeats": timing_repeats,
              "correctness_only": bool(args.correctness_only), "input_count": len(args.inputs),
              "fp32_output": bool(args.fp32_output),
              "cases": [], "timings": [], "layers": [],
              "rowwise_relative_l2_gate": ROWWISE_REL_GATE,
              "reconstruct_relative_l2_gate": RECONSTRUCT_REL_GATE,
              "dispatch_scope": "EXL3_SMALLM=1 native full batch versus rowwise ordinary GEMV; reconstruct=True packed reference. No compat wrapper; LinearEXL3.forward used directly.",
              "timing_scope": "Synchronized operator timing with CUDA events and monotonic wall; GPU-resident inputs/outputs; no .cpu()/validation/reductions in timed loops. Warmups excluded. Operator diagnostics, not end-to-end tok/s.",
              "operator_scope": "Packed-weight operator diagnostics; no BF16-fidelity claim."}
    log = (args.output / "events.jsonl").open("x", encoding="utf-8")

    def record(stage, **values):
        item = dict(stage=stage, elapsed_seconds=time.monotonic() - started, **values)
        log.write(json.dumps(item, allow_nan=False) + "\n")
        log.flush()
        print(json.dumps(item, allow_nan=False), flush=True)

    def memory():
        try:
            import torch as _torch
            return {"allocated_bytes": _torch.cuda.memory_allocated(0),
                    "reserved_bytes": _torch.cuda.max_memory_reserved(0),
                    "peak_allocated_bytes": _torch.cuda.max_memory_allocated(0),
                    "peak_reserved_bytes": _torch.cuda.max_memory_reserved(0)}
        except Exception:
            return None

    outputs = {}
    try:
        import torch
        import torch.utils.cpp_extension as cpp
        from safetensors.torch import load_file, save_file

        torch.set_num_threads(2)

        def no_build(*unused_args, **unused_kwargs):
            raise RuntimeError("C++ JIT build forbidden")

        cpp.load = cpp.load_inline = no_build
        binaries = list(Path(args.extension_dir).glob("exllamav3_ext*.so"))
        if len(binaries) != 1:
            raise RuntimeError("expected exactly one pinned native extension")
        binary = binaries[0]
        if digest(binary) != args.expected_extension_sha256:
            raise RuntimeError("extension differs from requested verified binary")
        spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
        ext = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3_ext"] = ext
        spec.loader.exec_module(ext)
        from quantlab.methods.exl3.optimizations import configure_native
        status['native_optimizations'] = configure_native(binary,
            smallm_kernel=args.smallm_kernel, head_warps=args.head_warps, native_smallm=True)
        sys.path.insert(0, str(args.source_dir))
        from exllamav3.modules.quant.exl3 import LinearEXL3

        if getattr(LinearEXL3.forward, "_quantlab_noncooperative", False):
            raise RuntimeError("compat wrapper installed; would bypass kernel under test")
        direct_forward = LinearEXL3.forward
        device = "cuda:0"
        props = torch.cuda.get_device_properties(0)
        if GPU_BUDGET_BYTES > props.total_memory:
            raise ValueError("Allocator budget exceeds device capacity")
        torch.cuda.set_per_process_memory_fraction(GPU_BUDGET_BYTES / props.total_memory)
        torch.cuda.reset_peak_memory_stats(0)
        faulthandler.dump_traceback_later(120, repeat=True)
        record("runtime", harness_sha256=hashlib.sha256(HARNESS_SOURCE).hexdigest(),
               extension_sha256=args.expected_extension_sha256, torch=torch.__version__,
               hip=torch.version.hip, device=props.name, exl3_smallm=os.environ.get("EXL3_SMALLM"),
               exl3_gemv=os.environ.get("EXL3_GEMV"), compat_wrapper=False)
        status.update({"device": props.name, "torch": torch.__version__,
                       "hip": str(torch.version.hip), "harness_sha256": hashlib.sha256(HARNESS_SOURCE).hexdigest(),
                       "extension_sha256": args.expected_extension_sha256})

        def make_inputs(batch, in_features):
            zeros = torch.zeros((batch, in_features), dtype=torch.float16)
            gen = torch.Generator().manual_seed(1234 + batch)
            gaussian = torch.randn((batch, in_features), generator=gen).half()
            pattern = (torch.arange(in_features) % 2 * 2 - 1).half()
            alternating = pattern.unsqueeze(0).expand(batch, in_features).contiguous()
            return {"zeros": zeros.contiguous(), "gaussian": gaussian.contiguous(),
                    "alternating": alternating.contiguous()}

        for layer_index, layer_dir in enumerate(args.inputs):
            layer_dir = Path(layer_dir)
            meta = json.loads((layer_dir / "encoding.json").read_text())
            key, bits, expected_sha = validate_encoding_meta(meta)
            artifact = layer_artifact(layer_dir)
            actual_sha = digest(artifact)
            if actual_sha != expected_sha:
                raise RuntimeError(f"packed input hash changed in {layer_dir}")
            packed = load_file(str(artifact), device="cpu")
            if "mcg" in packed:
                raise ValueError("small-M candidate does not support the mcg codebook")
            codebook = 2 if "mul1" in packed else 0
            if codebook not in status['native_optimizations']['smallm_codebooks']:
                raise ValueError("Verified binary does not support this small-M codebook")
            for required in ("trellis", "suh", "svh"):
                if required not in packed:
                    raise ValueError(f"packed layer missing {required} in {layer_dir}")
            in_features, out_features = infer_dimensions(
                int(packed["suh"].numel()), int(packed["svh"].numel()),
                tuple(packed["trellis"].shape), bits)
            gpu = {k: v.to(device) for k, v in packed.items()}
            linear = LinearEXL3(None, in_features, out_features, **gpu,
                               out_dtype=torch.float32 if args.fp32_output else torch.float16)
            layer_record = {"index": layer_index, "input": str(layer_dir),
                            "artifact_sha256": actual_sha, "key": key, "bits": bits, "codebook": codebook,
                            "in_features": in_features, "out_features": out_features,
                            "file_bytes": artifact.stat().st_size,
                            "tensor_bytes": sum(v.numel() * v.element_size() for v in packed.values())}
            status["layers"].append(layer_record)
            record("layer", **layer_record)
            for batch in batches:
                families = make_inputs(batch, in_features)
                for family in FAMILIES:
                    x_gpu = families[family].to(device)
                    native_gpu = direct_forward(linear, x_gpu, {})
                    torch.cuda.synchronize()
                    native = native_gpu.cpu()
                    if batch == 1:
                        rowwise_gpu = direct_forward(linear, x_gpu, {})
                    else:
                        rows = [direct_forward(linear, x_gpu[i:i + 1], {}) for i in range(batch)]
                        rowwise_gpu = torch.cat(rows, dim=0)
                    torch.cuda.synchronize()
                    rowwise = rowwise_gpu.cpu()
                    recon_gpu = direct_forward(linear, x_gpu, {"reconstruct": True}, torch.float32)
                    torch.cuda.synchronize()
                    recon = recon_gpu.cpu()
                    if tuple(native.shape) != (batch, out_features):
                        raise RuntimeError(f"native shape {tuple(native.shape)} mismatches {(batch, out_features)}")
                    vs_row = compare_pair(native, rowwise)
                    vs_recon = compare_pair(native, recon)
                    case = {"layer": layer_index, "batch": batch, "family": family,
                            "shape": [batch, out_features],
                            "native_vs_rowwise": vs_row, "native_vs_reconstruct": vs_recon}
                    status["cases"].append(case)
                    record("case", layer=layer_index, batch=batch, family=family,
                           native_vs_rowwise=vs_row, native_vs_reconstruct=vs_recon)
                    prefix = f"layer{layer_index}_batch{batch}_{family}"
                    outputs[f"{prefix}_native"] = native.contiguous()
                    outputs[f"{prefix}_rowwise"] = rowwise.contiguous()
                    outputs[f"{prefix}_reconstruct"] = recon.contiguous()
                    if not (vs_row["shape_match"] and vs_row["both_finite"]):
                        raise RuntimeError(f"native vs rowwise invalid (layer {layer_index} batch {batch} {family})")
                    if not (vs_recon["shape_match"] and vs_recon["both_finite"]):
                        raise RuntimeError(f"native vs reconstruct invalid (layer {layer_index} batch {batch} {family})")
                    if vs_row["relative_l2"] is None or vs_row["relative_l2"] > ROWWISE_REL_GATE:
                        raise RuntimeError(f"native vs rowwise relative L2 {vs_row['relative_l2']} exceeds {ROWWISE_REL_GATE}")
                    if vs_recon["relative_l2"] is None or vs_recon["relative_l2"] > RECONSTRUCT_REL_GATE:
                        raise RuntimeError(f"native vs reconstruct relative L2 {vs_recon['relative_l2']} exceeds {RECONSTRUCT_REL_GATE}")
                if args.correctness_only:
                    continue
                # Gaussian operator timing only; correctness tensors are off the timed path.
                x_gpu = families["gaussian"].to(device)
                row_views = [x_gpu[i:i + 1] for i in range(batch)]
                for _ in range(WARMUPS):
                    direct_forward(linear, x_gpu, {})
                    torch.cuda.synchronize()
                for _ in range(WARMUPS):
                    if batch == 1:
                        direct_forward(linear, x_gpu, {})
                    else:
                        torch.cat([direct_forward(linear, r, {}) for r in row_views], dim=0)
                    torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                native_mono, native_cuda, row_mono, row_cuda = [], [], [], []
                group_orders = []
                group_medians = []
                for group in range(TIMING_GROUPS):
                    order = ["native", "rowwise"] if group % 2 == 0 else ["rowwise", "native"]
                    group_orders.append(order)
                    per_method = {}
                    for method in order:
                        mono_samples, cuda_samples = [], []
                        for _ in range(timing_repeats):
                            torch.cuda.synchronize()
                            tick = time.monotonic()
                            start_event.record()
                            if method == "native":
                                direct_forward(linear, x_gpu, {})
                            elif batch == 1:
                                direct_forward(linear, x_gpu, {})
                            else:
                                torch.cat([direct_forward(linear, r, {}) for r in row_views], dim=0)
                            end_event.record()
                            torch.cuda.synchronize()
                            mono_samples.append(time.monotonic() - tick)
                            cuda_samples.append(start_event.elapsed_time(end_event) / 1000.0)
                        per_method[method] = {"mono": mono_samples, "cuda": cuda_samples}
                        if method == "native":
                            native_mono.extend(mono_samples)
                            native_cuda.extend(cuda_samples)
                        else:
                            row_mono.extend(mono_samples)
                            row_cuda.extend(cuda_samples)
                    group_medians.append({
                        "group": group, "order": order,
                        "native_mono_median_seconds": median_values(per_method["native"]["mono"]),
                        "rowwise_mono_median_seconds": median_values(per_method["rowwise"]["mono"]),
                        "native_cuda_median_seconds": median_values(per_method["native"]["cuda"]),
                        "rowwise_cuda_median_seconds": median_values(per_method["rowwise"]["cuda"])})
                timing = {"layer": layer_index, "batch": batch, "in_features": in_features,
                          "out_features": out_features, "warmups": WARMUPS,
                          "groups": TIMING_GROUPS, "repeats_per_group": timing_repeats,
                          "group_orders": group_orders, "group_medians": group_medians,
                          "native_mono_per_call_seconds": native_mono,
                          "rowwise_mono_per_call_seconds": row_mono,
                          "native_cuda_per_call_seconds": native_cuda,
                          "rowwise_cuda_per_call_seconds": row_cuda,
                          "native_mono_median_seconds": median_values(native_mono),
                          "rowwise_mono_median_seconds": median_values(row_mono),
                          "native_cuda_median_seconds": median_values(native_cuda),
                          "rowwise_cuda_median_seconds": median_values(row_cuda)}
                status["timings"].append(timing)
                record("timing", layer=layer_index, batch=batch,
                       native_mono_median_seconds=timing["native_mono_median_seconds"],
                       rowwise_mono_median_seconds=timing["rowwise_mono_median_seconds"],
                       native_cuda_median_seconds=timing["native_cuda_median_seconds"],
                       rowwise_cuda_median_seconds=timing["rowwise_cuda_median_seconds"])
        save_file(outputs, str(args.output / "outputs.safetensors"))
        status["status"] = "passed"
        status["outputs_sha256"] = digest(args.output / "outputs.safetensors")
        record("completed", allocator=memory(), status="passed",
               case_count=len(status["cases"]), timing_count=len(status["timings"]))
    except BaseException as exc:
        mark_failed(status, exc)
        try:
            (args.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        try:
            record("failed", error_type=type(exc).__name__, error=str(exc))
        except Exception:
            pass
        raise
    finally:
        if outputs and not (args.output / "outputs.safetensors").exists():
            # Preserve exact partial outputs when a numerical gate fails.
            save_file(outputs, str(args.output / "outputs.safetensors"))
            status["outputs_sha256"] = digest(args.output / "outputs.safetensors")
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        status["process_wall_seconds"] = time.monotonic() - started
        status["allocator"] = memory()
        try:
            (args.output / "result.json").write_text(
                json.dumps(status, indent=2, allow_nan=False), encoding="utf-8")
        except Exception:
            pass
        try:
            log.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
