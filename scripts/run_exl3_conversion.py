"""Run pinned EXL3 conversion with text/MTP source mapping and no C++ JIT.

An external runner must enforce wall time and host/GPU memory stop conditions.
Pass wrapper options before ``--`` and the verified upstream converter flags
after it. Calibration tokens must be explicitly supplied; no default corpus.
"""

import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib

from inspect_exl3_source import mapped_text_config


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(value):
    if sys.platform == "linux" and len(value) > 2 and value[1] == ":":
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    return Path(value)


def compact_quant_args(factory):
    """Set the opt-in before either single-linear or grouped conversion starts."""
    def create(*args, **kwargs):
        result = factory(*args, **kwargs)
        result["compact_cpu_buffers"] = True
        return result
    return create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--noncoop-compat", action="store_true", help="Experimental rowwise GEMV / reconstructed GEMM route")
    parser.add_argument("--compact-cpu-buffers", action="store_true", help="Opt into patched quantizer host-memory reduction")
    parser.add_argument("converter_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    local_config = tomllib.loads(args.config.read_text())
    if not args.execute or not all(local_config["execution"].get(k) is True for k in
                                   ("allow_local_inference", "allow_backend_probes")):
        parser.error("requires --execute and both configured local permissions")
    source = local_path(local_config["paths"]["bf16_model_dir"]).resolve()
    binary, = args.extension_dir.glob("exllamav3_ext*.so")
    if digest(binary) != args.expected_extension_sha256:
        parser.error("extension SHA256 differs from requested binary")
    if args.compact_cpu_buffers:
        for relative in ("exllamav3/modules/linear.py", "exllamav3/modules/quant/exl3_lib/quantize.py"):
            path = args.source_dir / relative
            if not path.is_file() or "compact_cpu_buffers" not in path.read_text():
                parser.error("--compact-cpu-buffers requires patched linear and quantizer sources")

    if args.noncoop_compat:
        os.environ["EXL3_BC_ATTN"] = "0"
        os.environ["EXL3_GEMV"] = "2"
    # The external supervisor owns timeouts. Background traceback dumps have
    # crashed in _Py_DumpTracebackThreads on WSL/Python 3.12; retain fatal traces.
    faulthandler.enable()
    import torch
    import torch.utils.cpp_extension as cpp

    def no_build(*unused_args, **unused_kwargs):
        raise RuntimeError("C++ JIT build forbidden")

    cpp.load = cpp.load_inline = no_build
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("exllamav3_ext", binary)
    extension = importlib.util.module_from_spec(spec)
    sys.modules["exllamav3_ext"] = extension
    spec.loader.exec_module(extension)
    sys.path.insert(0, str(args.source_dir))
    from exllamav3 import Config, Model
    from exllamav3.conversion import convert_model
    from safetensors.torch import load_file

    argv = args.converter_args
    if argv and argv[0] == "--":
        argv = argv[1:]
    options = convert_model.parser.parse_args(argv)
    stored = {}
    if options.resume:
        stored = json.loads((Path(options.work_dir) / "args.json").read_text())
        for key in ("bits", "head_bits", "mtp_bits", "codebook", "cal_data", "cal_rows", "cal_cols"):
            proposed = getattr(options, key)
            if proposed is not None and proposed != stored.get(key):
                parser.error(f"resume cannot change {key}; pass the original settings")
    if options.override_anyway:
        parser.error("--override_anyway is disabled; changed treatments require a fresh run")
    input_dir = Path(options.in_dir or stored.get("in_dir", "")).resolve()
    output_dir = Path(options.out_dir or stored.get("out_dir", "")).resolve()
    work_dir = Path(options.work_dir or "").resolve()
    if input_dir != source:
        parser.error("converter input must match configured BF16 source")
    if not options.work_dir or not (options.out_dir or stored.get("out_dir")):
        parser.error("explicit working and output directories required")
    paths = (input_dir, output_dir, work_dir)
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        parser.error("source, work and output directories must be separate, nonnested paths")
    if not options.resume:
        for path in (work_dir, output_dir):
            if path.exists() and any(path.iterdir()):
                parser.error("fresh conversion requires empty or new work/output directories")
    calibration = options.cal_data or stored.get("cal_data")
    if not calibration:
        parser.error("explicit -cd calibration token file required; default corpus disabled")
    rows = options.cal_rows or stored.get("cal_rows")
    cols = options.cal_cols or stored.get("cal_cols")
    if not rows or not cols or rows <= 0 or cols <= 0:
        parser.error("explicit positive -cr and -cc required")
    tokens = load_file(calibration, device="cpu")["input_ids"]
    vocab = json.loads((source / "config.json").read_text())["text_config"]["vocab_size"]
    if (tokens.dtype != torch.int64 or tokens.ndim != 2 or tokens.shape[0] < rows
            or tokens.shape[1] < cols or tokens.min().item() < 0 or tokens.max().item() >= vocab):
        parser.error("calibration must contain valid int64 input_ids with enough rows/columns")
    del tokens
    torch.cuda.set_per_process_memory_fraction(10 * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    original = Config.from_directory
    original_model = Model.from_config
    original_quant_args = convert_model.make_quant_args
    if args.noncoop_compat:
        from quantlab.methods.exl3.compat import install, prepare_loaded_module

    def mapped_from_directory(directory, **kwargs):
        if Path(directory).resolve() in (input_dir, output_dir):
            if kwargs:
                raise ValueError("mapped text config does not support extra config options")
            config = mapped_text_config(directory)[0]
            if args.noncoop_compat:
                install(config)
                config._compat_top_modules = []
                end_deferred = config.stc.end_deferred_load
                def end_deferred_and_prepare():
                    result = end_deferred()
                    for module in config._compat_top_modules:
                        if module.device is not None:
                            prepare_loaded_module(module)
                    return result
                config.stc.end_deferred_load = end_deferred_and_prepare
            return config
        return original(directory, **kwargs)

    def mapped_model(config, *model_args, **kwargs):
        model = original_model(config, *model_args, **kwargs)
        if args.noncoop_compat and hasattr(config, "_compat_top_modules"):
            for module in model.modules:
                config._compat_top_modules.append(module)
                load = module.load
                def load_and_prepare(*load_args, _load=load, _module=module, **load_kwargs):
                    result = _load(*load_args, **load_kwargs)
                    prepare_loaded_module(_module)
                    return result
                module.load = load_and_prepare
        return model

    # compile_model's quantization_config.json generation reopens the output
    # through this same class. Hook both paths; the direct mapped constructor
    # does not call from_directory, so there is no recursion.
    Config.from_directory = staticmethod(mapped_from_directory)
    Model.from_config = staticmethod(mapped_model)
    if args.compact_cpu_buffers:
        convert_model.make_quant_args = compact_quant_args(original_quant_args)
    try:
        print(json.dumps({"extension_sha256": digest(binary), "calibration_sha256": digest(calibration),
                          "converter_argv": argv, "text_only": True, "mtp_included": True,
                          "experimental_noncoop_compat": args.noncoop_compat,
                          "compact_cpu_buffers": args.compact_cpu_buffers}), flush=True)
        prepared, state, ok, error = convert_model.prepare(options)
        if not ok:
            raise ValueError(error)
        convert_model.main(prepared, state)
    finally:
        Config.from_directory = staticmethod(original)
        Model.from_config = staticmethod(original_model)
        convert_model.make_quant_args = original_quant_args


if __name__ == "__main__":
    main()
