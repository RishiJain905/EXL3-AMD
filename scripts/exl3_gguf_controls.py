"""Bounded Q2_K/Q3_K CPU controls through the configured ggml-base DLL.

Uses inspected ggml-quants.h C signatures and ggml-common.h block layouts.
No model execution, build, install, GPU initialization or backend modification.
The output GGUF files are single-tensor diagnostic artifacts, not full models.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import types

from run_exl3_layer import FAMILIES


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def nmse(actual, reference):
    return float(((actual - reference).square().sum() / reference.square().sum()).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if not args.execute or not all(config["execution"].get(k) is True for k in ("allow_backend_probes", "allow_local_inference")):
        parser.error("requires --execute and both configured local permissions")
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(2)
    backend = Path(config["paths"]["llama_cpp_root"])
    # The protected checkout's package __init__ imports an inconsistent MTP
    # tensor-name map. Load only its standalone format modules without editing it.
    package = types.ModuleType("gguf")
    package.__path__ = [str(backend / "gguf-py/gguf")]
    sys.modules["gguf"] = package
    from gguf.gguf_reader import GGUFReader
    from gguf.gguf_writer import GGUFWriter
    from gguf.constants import GGMLQuantizationType

    dll_path = Path(config["paths"]["llama_bin_dir"]) / "ggml-base.dll"
    dll = ctypes.CDLL(str(dll_path))
    dll.ggml_type_size.argtypes = [ctypes.c_int]
    dll.ggml_type_size.restype = ctypes.c_size_t
    modes = {}
    for bits, size in ((2, 84), (3, 110)):
        kind = getattr(GGMLQuantizationType, f"Q{bits}_K")
        if dll.ggml_type_size(int(kind)) != size:
            raise RuntimeError("DLL block size differs from inspected source")
        encode = getattr(dll, f"quantize_row_q{bits}_K_ref")
        decode = getattr(dll, f"dequantize_row_q{bits}_K")
        for fn in (encode, decode):
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fn.restype = None
        modes[bits] = (size, kind, encode, decode)

    if args.reload:
        report = json.loads((args.output / "encoding.json").read_text())
        if report["dll_sha256"] != digest(dll_path):
            raise RuntimeError("DLL changed since encoding")
        rows = []
        x = torch.randn(128, 256, generator=torch.Generator().manual_seed(200)).half().float()
        for item in report["artifacts"]:
            path = args.output / item["file"]
            if digest(path) != item["sha256"]:
                raise RuntimeError("GGUF changed since encoding")
            reader = GGUFReader(path)
            tensor, = reader.tensors
            size, kind, _, decode = modes[item["bits"]]
            if tensor.tensor_type != kind or tuple(tensor.shape) != (256, 128):
                raise RuntimeError("reloaded tensor metadata mismatch")
            packed = np.ascontiguousarray(tensor.data, dtype=np.uint8)
            if hashlib.sha256(packed.tobytes()).hexdigest() != item["payload_sha256"]:
                raise RuntimeError("reloaded packed bytes mismatch")
            decoded = np.empty((128, 256), dtype=np.float32)
            decode(packed.ctypes.data, decoded.ctypes.data, decoded.size)
            if not np.isfinite(decoded).all():
                raise RuntimeError("nonfinite reconstruction")
            source_path = args.output / f'{item["family"]}-source.safetensors'
            if digest(source_path) != item["source_safetensors_sha256"]:
                raise RuntimeError("source artifact changed")
            ref = load_file(str(source_path))["weight_bf16"].float()
            weights = torch.from_numpy(decoded)
            row = dict(**item, weight_nmse=nmse(weights, ref), output_nmse={str(batch): nmse(x[:batch] @ weights.T, x[:batch] @ ref.T) for batch in (1, 16, 128)}, reload_payload_exact=True)
            rows.append(row)
        result = dict(hardware="CPU", torch=torch.__version__, synthetic_input="torch.randn(128,256), CPU generator seed 200, cast fp16 then fp32 to match EXL3 probe", artifacts=rows)
        with (args.output / "results.json").open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        print(json.dumps(result, allow_nan=False))
        return

    args.output.mkdir(parents=True, exist_ok=False)
    model = Path(config["paths"]["bf16_model_dir"])
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    records = []
    for family, suffix in FAMILIES.items():
        keys = [key for key in index if key.startswith("model.") and key.endswith(suffix)]
        if len(keys) != 1:
            raise RuntimeError("source selection not unique")
        key = keys[0]
        with safe_open(model / index[key], framework="pt", device="cpu") as f:
            source = f.get_slice(key)[:128, :256].contiguous()
        if source.dtype != torch.bfloat16 or tuple(source.shape) != (128, 256):
            raise RuntimeError("expected exact BF16 slice")
        source_path = args.output / f"{family}-source.safetensors"
        save_file({"weight_bf16": source}, str(source_path))
        floats = source.float().numpy()
        for bits, (size, kind, encode, _) in modes.items():
            packed = np.empty((128, size), dtype=np.uint8)
            start = time.perf_counter()
            encode(floats.ctypes.data, packed.ctypes.data, floats.size)
            elapsed = time.perf_counter() - start
            path = args.output / f"{family}-q{bits}_k.gguf"
            writer = GGUFWriter(path, "quantlab_slice")
            writer.add_tensor("weight", packed, raw_dtype=kind)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            records.append(dict(family=family, key=key, bits=bits, file=path.name,
                                payload_bytes=packed.nbytes, file_bytes=path.stat().st_size,
                                container_overhead_bytes=path.stat().st_size-packed.nbytes,
                                payload_bpw=packed.nbytes*8/floats.size,
                                file_bpw=path.stat().st_size*8/floats.size,
                                source_safetensors_sha256=digest(source_path),
                                source_bf16_bytes_sha256=hashlib.sha256(source.view(torch.uint16).numpy().tobytes()).hexdigest(),
                                sha256=digest(path), payload_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
                                encode_seconds=elapsed))
    provenance_files = ["ggml/src/ggml-quants.h", "ggml/src/ggml-quants.c", "ggml/src/ggml-common.h", "ggml/include/ggml.h", "gguf-py/gguf/gguf_writer.py", "gguf-py/gguf/gguf_reader.py"]
    report = dict(dll_sha256=digest(dll_path), source_sha256={name: digest(backend / name) for name in provenance_files}, index_sha256=digest(index_path), artifacts=records)
    with (args.output / "encoding.json").open("x") as f:
        json.dump(report, f, indent=2)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()), "--output", str(args.output.resolve()), "--reload", "--execute"], check=True, timeout=120)


if __name__ == "__main__":
    main()
