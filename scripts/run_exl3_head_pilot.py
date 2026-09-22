"""Full production-shaped head memory pilot; external monitoring is mandatory.

Synthetic calibration tests memory/runtime compatibility, not model fidelity.
No package/build actions. Encode writes only packed weights and provenance.
"""

import argparse
import ast
import faulthandler
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_exl3_layer import digest, local_path

SHAPE = (248320, 5120)


def tensor_provenance(shard, key, shape=SHAPE):
    with shard.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 16 * 2**20:
            raise ValueError("unexpected safetensors header size")
        header = stream.read(size)
        entry = json.loads(header)[key]
        if tuple(entry["shape"]) != tuple(shape) or entry["dtype"] != "BF16":
            raise ValueError("head shape/dtype differs from preregistered source")
        begin, end = entry["data_offsets"]
        if end - begin != shape[0] * shape[1] * 2:
            raise ValueError("head tensor byte range is inconsistent")
        stream.seek(8 + size + begin)
        remaining = end - begin
        checksum = hashlib.sha256()
        while remaining:
            data = stream.read(min(8 * 2**20, remaining))
            if not data:
                raise ValueError("truncated head tensor")
            checksum.update(data)
            remaining -= len(data)
    return dict(shard_name=shard.name, shard_bytes=shard.stat().st_size,
                shard_sha256=digest(shard), header_sha256=hashlib.sha256(header).hexdigest(),
                tensor_sha256=checksum.hexdigest(), tensor_header=entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("encode", "reload"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--shape", type=int, nargs=2, default=SHAPE, metavar=("OUTPUTS", "INPUTS"))
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 5, 6), default=4)
    parser.add_argument("--codebook", choices=("3inst", "mul1"), default="3inst")
    parser.add_argument("--out-scales", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    shape, bits = tuple(args.shape), args.bits
    mul1 = args.codebook == "mul1"
    codebook = 2 if mul1 else 0
    if min(shape) <= 0 or any(size % 128 for size in shape):
        parser.error("head dimensions must be positive multiples of 128")
    config = tomllib.loads(args.config.read_text())
    if not args.execute or not all(config["execution"].get(k) is True for k in
                                  ("allow_local_inference", "allow_backend_probes")):
        parser.error("requires --execute and both configured local permissions")
    if args.mode == "reload" and args.input is None:
        parser.error("reload requires --input")
    quantizer = args.source_dir / "exllamav3/modules/quant/exl3_lib/quantize.py"
    functions = {n.name for n in ast.parse(quantizer.read_text()).body if isinstance(n, ast.FunctionDef)}
    if not {"block_proxy_error_streamed", "ldlq_packed_columns"} <= functions:
        parser.error("source requires the reviewed head-memory-v5.patch")
    binary, = args.extension_dir.glob("exllamav3_ext*.so")
    if digest(binary) != args.expected_extension_sha256:
        parser.error("extension hash differs from requested binary")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    log = (args.output / "events.jsonl").open("x")
    faulthandler.dump_traceback_later(120, repeat=True)

    def record(stage, **values):
        line = json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-started, **values), allow_nan=False)
        log.write(line + "\n"); log.flush()
        print(line, flush=True)

    try:
        import torch
        import torch.utils.cpp_extension as cpp
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        torch.set_num_threads(2)

        def no_build(*unused, **kwargs):
            raise RuntimeError("C++ JIT build forbidden")
        cpp.load = cpp.load_inline = no_build
        spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
        ext = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3_ext"] = ext
        spec.loader.exec_module(ext)
        sys.path.insert(0, str(args.source_dir))
        from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
        device = "cuda:0"
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(10 * 2**30 / props.total_memory)
        torch.cuda.reset_peak_memory_stats()
        record("runtime", device=props.name, torch=torch.__version__, hip=torch.version.hip,
               extension_sha256=digest(binary), quantizer_sha256=digest(quantizer),
               linear_sha256=digest(args.source_dir / "exllamav3/modules/linear.py"),
               harness_sha256=digest(Path(__file__)), mode=args.mode,
               scope="memory/operator compatibility; synthetic calibration; no quality selection")

        if args.mode == "encode":
            model = local_path(config["paths"]["bf16_model_dir"])
            index_file = model / "model.safetensors.index.json"
            index = json.loads(index_file.read_text())["weight_map"]
            key = "lm_head.weight"
            shard = model / index[key]
            provenance = tensor_provenance(shard, key, shape)
            record("source", key=key, shape=shape, index_sha256=digest(index_file), **provenance)
            with safe_open(shard, framework="pt", device="cpu") as file:
                original = file.get_tensor(key)
                if original.dtype != torch.bfloat16 or tuple(original.shape) != shape:
                    raise ValueError("expected full BF16 head")
                # Match upstream's FP16 checkpoint loading without retaining a
                # full FP32 original. No source-weight copy is written to disk.
                weight_holder = [original.half().T.contiguous()]
                del original
            del file
            cal = torch.randn(512, shape[1], generator=torch.Generator().manual_seed(100))
            h = {"H": (cal.T @ cal).to(device), "count": 512, "finalized": False,
                 "device": device, "first_key": key}
            del cal
            qa = {"K": bits, "seed": args.seed,
                  "apply_out_scales": {"auto": None, "always": True, "never": False}[args.out_scales],
                  "devices": [device], "compact_cpu_buffers": True, "memory_diagnostics": True,
                  "mul1": mul1, "sigma_reg": 0.025}
            begin = time.monotonic()
            # Transfer sole ownership: the encoder can release the CPU source
            # once its synchronous upload has completed, before CPU LDLQ buffers.
            returned, proxy, packed = quantize_exl3(weight_holder.pop(), h, qa, False, verbose=False)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            if returned is not None or not math.isfinite(proxy) or proxy < 0:
                raise RuntimeError("compact quantization returned unexpected weight or failed proxy")
            del weight_holder, h
            packed = {name: value.cpu().contiguous() for name, value in packed.items()}
            if any(not torch.isfinite(value).all() for value in packed.values() if value.is_floating_point()):
                raise RuntimeError("nonfinite packed scales")
            save_file(packed, str(args.output / "head.safetensors"))
            artifact = args.output / "head.safetensors"
            if artifact.stat().st_size > 2**30:
                raise RuntimeError("packed artifact exceeds 1 GiB pilot bound")
            result = dict(bits=bits, codebook=args.codebook, shape=shape, key=key,
                          source_provenance=provenance, source_index_sha256=digest(index_file),
                          quant_args=qa, calibration=dict(rows=512, columns=shape[1], seed=100, kind="synthetic Gaussian"),
                          source_conversion="BF16 source -> FP16 host -> FP32 GPU arithmetic",
                          artifact_sha256=digest(artifact), file_bytes=artifact.stat().st_size,
                          tensor_bytes=sum(v.numel()*v.element_size() for v in packed.values()),
                          conversion_seconds=elapsed, proxy_error=proxy)
            (args.output / "encoding.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            record("encoded", **result)
        else:
            from quantlab.methods.exl3.oracle import decode_trellis, reconstruct
            artifact = args.input / "head.safetensors"
            metadata = json.loads((args.input / "encoding.json").read_text())
            if metadata["bits"] != bits or metadata["codebook"] != args.codebook or tuple(metadata["shape"]) != shape or digest(artifact) != metadata["artifact_sha256"]:
                raise ValueError("packed head provenance mismatch")
            packed = load_file(str(artifact), device="cpu")
            gpu = {name: value.to(device) for name, value in packed.items()}
            native = torch.empty((shape[1], shape[0]), dtype=torch.half, device=device)
            ext.reconstruct(native, gpu["trellis"], bits, False, mul1)
            torch.cuda.synchronize()
            for start in (0, (shape[0] // 2 // 128) * 128, shape[0] - 128):
                end = start + 128
                trellis = packed["trellis"][:, start // 16:end // 16, :].numpy()
                decoded = torch.from_numpy(decode_trellis(trellis, bits, codebook=codebook))
                actual = native[:, start:end].float().cpu()
                mismatches = int((decoded != actual).sum().item())
                intended = torch.from_numpy(reconstruct(trellis, bits, packed["suh"].numpy(), packed["svh"][start:end].numpy(), codebook=codebook))
                transformed = torch.empty((shape[1], 128), dtype=torch.half, device=device)
                ext.reconstruct_had_slice(transformed, gpu["trellis"], gpu["suh"], gpu["svh"][start:end], bits, False, mul1, start)
                actual_transformed = transformed.float().cpu()
                relative_rms = (((actual_transformed-intended).square().sum() / intended.square().sum().clamp_min(1e-20)).sqrt().item())
                record("sample_reload", output_columns=[start, end], values=decoded.numel(),
                       exact_decode_mismatches=mismatches, transformed_relative_rms=relative_rms)
                if mismatches or not torch.isfinite(actual_transformed).all() or relative_rms > 0.005:
                    raise RuntimeError("packed head sample reload mismatch")
            record("reloaded", artifact_sha256=metadata["artifact_sha256"], sampled_columns=384,
                   scope="first/middle/last 128-column groups; not exhaustive whole-head verification")
        record("completed", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        record("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        log.close()


if __name__ == "__main__":
    main()
