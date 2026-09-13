"""Encode or fresh-reload a bounded native EXL3 source slice. Use an external timeout.

Requires the a compatible, explicitly configured native runtime; never installs or builds.
Synthetic activation errors are operator diagnostics, not model fidelity.
"""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
HARNESS_SOURCE = Path(__file__).read_bytes()
sys.path.insert(0, str(ROOT / "src"))
PIN = "9d4c311204262bb3e64a1a69da7a9c9d24353d523bb226a39c926d8311cc6f01"
FAMILIES = {
    "mlp": "layers.0.mlp.up_proj.weight",
    "attention": "layers.3.self_attn.q_proj.weight",
    "linear_attention": "layers.0.linear_attn.out_proj.weight",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(value):
    if sys.platform == "linux" and len(value) > 2 and value[1] == ":":
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    return Path(value)


def require_compact_source(source):
    quantizer = Path(source) / "exllamav3/modules/quant/exl3_lib/quantize.py"
    if not quantizer.is_file() or "compact_cpu_buffers" not in quantizer.read_text():
        raise ValueError("--compact-cpu-buffers requires the patched quantizer source")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--extension-dir", type=Path)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--expected-extension-sha256", default=PIN)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["encode", "reload"], required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--family", choices=FAMILIES, default="mlp")
    p.add_argument("--bits", type=int, choices=[2, 3], default=3)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--columns", type=int, default=256)
    p.add_argument("--force-num-sms", type=int, choices=[1], help="Diagnostic packed dispatch bypassing cooperative autotune")
    p.add_argument("--rowwise", action="store_true", help="Diagnostic batches through native single-row packed GEMV")
    p.add_argument("--reconstruct", action="store_true", help="Diagnostic native per-layer dequantization plus hipBLAS")
    p.add_argument("--compact-cpu-buffers", action="store_true", help="Opt into patched quantizer CPU buffers; encode only")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.compact_cpu_buffers and a.mode != "encode":
        p.error("--compact-cpu-buffers is only valid with --mode encode")
    if any(v <= 0 or v % 128 for v in (a.rows, a.columns)) or a.rows * a.columns > 100_000_000:
        p.error("slice dimensions must be positive multiples of 128 with <=100 million elements")
    config = tomllib.loads(a.config.read_text())
    if not a.execute or not all(config["execution"].get(k) is True for k in
            ["allow_local_inference", "allow_backend_probes"]):
        p.error("requires --execute and both configured local permissions")
    if a.mode == "reload" and not a.input:
        p.error("reload requires --input")
    source = a.source_dir or a.runtime / "rocm-exl3"
    if a.compact_cpu_buffers:
        require_compact_source(source)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "harness.py").write_bytes(HARNESS_SOURCE)
    faulthandler.dump_traceback_later(120, repeat=True)
    started = time.monotonic()
    log = (a.output / "events.jsonl").open("x")

    def record(stage, **values):
        item = dict(stage=stage, elapsed_seconds=time.monotonic() - started, **values)
        log.write(json.dumps(item, allow_nan=False) + "\n")
        log.flush()
        print(json.dumps(item, allow_nan=False), flush=True)

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        import torch.utils.cpp_extension as cpp

        torch.set_num_threads(2)
        def no_build(*args, **kwargs):
            raise RuntimeError("C++ JIT build forbidden")
        cpp.load = cpp.load_inline = no_build
        binary, = (a.extension_dir or a.runtime / "full-build-v4/lib").glob("exllamav3_ext*.so")
        if digest(binary) != a.expected_extension_sha256:
            raise RuntimeError("extension differs from requested verified binary")
        spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
        ext = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3_ext"] = ext
        spec.loader.exec_module(ext)
        sys.path.insert(0, str(source))
        from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
        from exllamav3.modules.quant.exl3 import LinearEXL3
        device = "cuda:0"
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(10 * 2**30 / props.total_memory)
        torch.cuda.reset_peak_memory_stats()
        record("runtime", harness_sha256=hashlib.sha256(HARNESS_SOURCE).hexdigest(),
               extension_sha256=digest(binary), torch=torch.__version__, hip=torch.version.hip,
               device=props.name, quantizer_sha256=digest(source / "exllamav3/modules/quant/exl3_lib/quantize.py"))
        if a.mode == "encode":
            model = local_path(config["paths"]["bf16_model_dir"])
            index_file = model / "model.safetensors.index.json"
            index = json.loads(index_file.read_text())["weight_map"]
            keys = [k for k in index if k.startswith("model.language_model.") and k.endswith(FAMILIES[a.family])]
            if len(keys) != 1:
                raise ValueError(f"source selection not unique: {keys}")
            key = keys[0]
            with safe_open(model / index[key], framework="pt", device="cpu") as f:
                sl = f.get_slice(key)
                shape = sl.get_shape()
                original = sl[:a.rows, :a.columns].contiguous()
            if original.dtype != torch.bfloat16 or original.shape != (a.rows, a.columns):
                raise ValueError("expected BF16 source and exact preregistered slice")
            w = original.float().T.contiguous()
            save_file({"weight_bf16": original}, str(a.output / "source.safetensors"))
            record("source", key=key, full_shape=shape, slice_shape=list(original.shape),
                   source_slice_sha256=digest(a.output / "source.safetensors"), index_sha256=digest(index_file))
            gen = torch.Generator().manual_seed(100)
            cal = torch.randn(512, a.columns, generator=gen)
            h = {"H": (cal.T @ cal).to(device), "count": 512, "finalized": False,
                 "device": device, "first_key": key}
            qa = {"K": a.bits, "seed": 20260909, "apply_out_scales": None, "devices": [device]}
            if a.compact_cpu_buffers:
                qa["compact_cpu_buffers"] = True
            start = time.monotonic()
            q, proxy, packed = quantize_exl3(w if a.compact_cpu_buffers else w.to(device), h, qa, True)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            packed = {k: v.cpu().contiguous() for k, v in packed.items()}
            save_file(packed, str(a.output / "layer.safetensors"))
            save_file({"weight_q": q.cpu().contiguous()}, str(a.output / "intended.safetensors"))
            meta = dict(family=a.family, bits=a.bits, key=key, quant_args=qa,
                        tensor_bytes=sum(v.numel()*v.element_size() for v in packed.values()),
                        file_bytes=(a.output / "layer.safetensors").stat().st_size,
                        artifact_sha256=digest(a.output / "layer.safetensors"),
                        conversion_seconds=elapsed, proxy_error=proxy)
            (a.output / "encoding.json").write_text(json.dumps(meta, indent=2))
            record("encoded", **meta)
        else:
            import numpy as np
            from quantlab.methods.exl3.oracle import decode_trellis, reconstruct
            packed = load_file(str(a.input / "layer.safetensors"))
            meta = json.loads((a.input / "encoding.json").read_text())
            if digest(a.input / "layer.safetensors") != meta["artifact_sha256"]:
                raise RuntimeError("packed input hash changed")
            bits = meta["bits"]
            arrays = {k: v.numpy() for k,v in packed.items()}
            raw = decode_trellis(arrays["trellis"], bits)
            independent = reconstruct(arrays["trellis"], bits, arrays["suh"], arrays["svh"])
            gpu = {k:v.to(device) for k,v in packed.items()}
            native = torch.empty(raw.shape, dtype=torch.half, device=device)
            ext.reconstruct(native, gpu["trellis"], bits, False, False)
            torch.cuda.synchronize()
            mismatch = int(np.count_nonzero(native.float().cpu().numpy() != raw))
            record("independent_decode", values=int(raw.size), exact_mismatches=mismatch)
            if mismatch:
                raise RuntimeError("independent versus native decode mismatch")
            def nmse(actual, ref):
                return float(((actual.float()-ref.float()).square().sum() / ref.float().square().sum()).item())
            ref = load_file(str(a.input / "source.safetensors"))["weight_bf16"].float().T.contiguous()
            intended = load_file(str(a.input / "intended.safetensors"))["weight_q"]
            iw = torch.from_numpy(independent)
            record("reconstruction", weight_nmse=nmse(iw,ref), intended_nmse=nmse(iw,intended))
            if nmse(iw,intended)**0.5 > 0.005:
                raise RuntimeError("packed reconstruction differs from encoder")
            linear = LinearEXL3(None, ref.shape[0], ref.shape[1], **gpu, out_dtype=torch.float16)
            x = torch.randn(128, ref.shape[0], generator=torch.Generator().manual_seed(200)).half()
            outputs = {}
            for batch in ((2, 3, 16, 128) if a.reconstruct else (1, 16, 128)):
                xx = x[:batch].contiguous()
                if a.reconstruct:
                    actual = linear.forward(xx.to(device), {"reconstruct": True}, torch.float32).cpu()
                elif a.rowwise:
                    actual = torch.cat([linear.forward(row[None].to(device), {}).cpu() for row in xx])
                elif a.force_num_sms:
                    x_gpu = xx.to(device)
                    y_gpu = torch.empty((batch, ref.shape[1]), dtype=torch.half, device=device)
                    ext.exl3_gemm(x_gpu, gpu["trellis"], y_gpu, gpu["suh"],
                                  torch.empty_like(x_gpu), gpu["svh"], 0, False, True, a.force_num_sms)
                    actual = y_gpu.cpu()
                else:
                    actual = linear.forward(xx.to(device), {}).cpu()
                torch.cuda.synchronize()
                expected = xx.float() @ iw
                source_ref = xx.float() @ ref
                fp16_floor = (xx.to(device) @ ref.half().to(device)).cpu()
                error = nmse(actual,expected)
                record("forward", batch=batch, force_num_sms=a.force_num_sms, rowwise=a.rowwise,
                       reconstruct=a.reconstruct, packed_reference_nmse=error,
                       source_bf16_nmse=nmse(actual,source_ref), fp16_floor_nmse=nmse(fp16_floor,source_ref))
                outputs[f"batch_{batch}"] = actual
                if not torch.isfinite(actual).all() or error**0.5 > 0.005:
                    raise RuntimeError("native packed forward mismatch")
            save_file(outputs, str(a.output / "outputs.safetensors"))
        record("completed", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(), status="passed")
    except Exception as exc:
        record("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        log.close()


if __name__ == "__main__":
    main()
