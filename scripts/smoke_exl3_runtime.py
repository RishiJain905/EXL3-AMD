"""Bounded precompiled EXL3 smoke; caller must enforce a 60-second timeout.

No model loading, compilation, or installation. Tensor allocations are tiny;
the 256 MiB PyTorch allocator limit does not cap driver/import overhead.
Output is append-only JSONL so a timeout preserves completed stages.
"""

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import platform
import struct
import sys
import tomllib
import traceback


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def decode_scalar(state):
    """QTIP 3inst: uint32 MAD, (x & mask) XOR bias, rounded half sum.

    Independent scalar implementation of cb=0 in quant/codebook.cuh;
    lop3 LUT 0x6a implements (a & b) ^ c.
    """
    word = ((state * 89226354 + 64248484) & 0xFFFFFFFF)
    word = (word & 0x8FFF8FFF) ^ 0x3B603B60
    low, high = struct.unpack("<ee", struct.pack("<I", word))
    return struct.unpack("<e", struct.pack("<e", low + high))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--extension-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--decode", action="store_true", help="Check all 65536 cb=0 states")
    parser.add_argument("--import-package", action="store_true", help="Also import pinned exllamav3")
    args = parser.parse_args()
    # Exclusive creation rejects a reused output before any third-party import.
    with args.output.open("x", encoding="utf-8") as output:
        def record(stage, **values):
            output.write(json.dumps({"stage": stage, **values}, allow_nan=False) + "\n")
            output.flush()

        record("started", python=platform.python_version(), status="running")
        try:
            with args.config.open("rb") as handle:
                execution = tomllib.load(handle).get("execution", {})
            if not args.execute or not all(execution.get(key) is True for key in
                    ("allow_backend_probes", "allow_local_inference")):
                raise RuntimeError("Requires --execute and both local execution permissions")
            source = args.source_dir.resolve(strict=True)
            extension_dir = args.extension_dir.resolve(strict=True)
            if not (source / "exllamav3" / "__init__.py").is_file():
                raise RuntimeError("Source directory must contain the pinned exllamav3 package")
            candidates = [extension_dir / ("exllamav3_ext" + suffix)
                          for suffix in importlib.machinery.EXTENSION_SUFFIXES
                          if (extension_dir / ("exllamav3_ext" + suffix)).is_file()]
            if len(candidates) != 1:
                raise RuntimeError("Expected exactly one compatible precompiled extension")
            binary = candidates[0].resolve(strict=True)
            if binary.parent != extension_dir:
                raise RuntimeError("Extension resolves outside requested directory")

            import torch

            # Load only the exact compiled file; do not import the package's JIT loader.
            spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
            if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
                raise RuntimeError("Expected native extension loader")
            ext = importlib.util.module_from_spec(spec)
            sys.modules["exllamav3_ext"] = ext
            spec.loader.exec_module(ext)
            if Path(ext.__file__).resolve() != binary:
                raise RuntimeError("Unexpected imported extension origin")
            record("extension", sha256=sha256(binary), filename=binary.name,
                   torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda)
            if not torch.cuda.is_available():
                raise RuntimeError("GPU unavailable")
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(device)
            limit = 256 * 1024 * 1024
            torch.cuda.set_per_process_memory_fraction(min(1.0, limit / props.total_memory), device)
            torch.cuda.reset_peak_memory_stats(device)
            record("device", name=props.name, total_memory_bytes=props.total_memory,
                   allocator_limit_bytes=limit, driver_memory_cap=False)

            with torch.inference_mode():
                x_cpu = ((torch.arange(1024, dtype=torch.float32) % 37 - 18) / 16).reshape(4, 256).half()
                w_cpu = (torch.arange(256, dtype=torch.float32) % 17 / 32 + 0.5).half()
                reference = (x_cpu.float() * w_cpu.float() *
                             torch.rsqrt(x_cpu.float().square().mean(-1, keepdim=True) + 1e-5)).half()
                x, w = x_cpu.to(device), w_cpu.to(device)
                y = torch.empty_like(x)
                ext.rms_norm(x, w, y, 1e-5, 0.0, 1.0, False, False)
                torch.cuda.synchronize(device)
                actual = y.cpu()
                mismatch = int((~torch.isclose(actual, reference, rtol=1e-3, atol=1e-3)).sum())
                error = (actual.float() - reference.float()).abs().max().item()
                record("rms_norm", elements=actual.numel(), mismatches=mismatch,
                       max_abs_error=error if torch.isfinite(actual).all() else None, rtol=1e-3, atol=1e-3)
                if mismatch:
                    raise RuntimeError("RMS norm mismatch")
                if args.decode:
                    states = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256).to(device)
                    decoded = torch.empty((256, 256), dtype=torch.float32, device=device)
                    reference = torch.tensor([decode_scalar(i) for i in range(65536)], dtype=torch.float32).reshape(256, 256)
                    ext.decode(states, decoded, False, False)
                    torch.cuda.synchronize(device)
                    actual = decoded.cpu()
                    mismatch = int((actual != reference).sum())
                    record("decode_3inst", elements=65536, mismatches=mismatch,
                           max_abs_error=(actual - reference).abs().max().item() if torch.isfinite(actual).all() else None)
                    if mismatch:
                        raise RuntimeError("3inst decode mismatch")

            record("memory", peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                   driver_overhead_bytes=None)

            if args.import_package:
                # The package sees the compiled module in sys.modules. Also disable
                # torch JIT entry points explicitly so any unexpected fallback fails.
                import torch.utils.cpp_extension as cpp_extension

                def no_jit(*unused_args, **unused_kwargs):
                    raise RuntimeError("JIT compilation is forbidden in this smoke")

                cpp_extension.load = no_jit
                cpp_extension.load_inline = no_jit
                sys.path.insert(0, str(source))
                package = importlib.import_module("exllamav3")
                if Path(package.__file__).resolve() != (source / "exllamav3" / "__init__.py").resolve():
                    raise RuntimeError("Unexpected package origin")
                record("package_import", status="passed", version=getattr(package, "__version__", None))
            record("completed", status="passed")
            return 0
        except Exception as exc:
            record("failed", status="failed", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
