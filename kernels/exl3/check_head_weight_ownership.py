"""CPU checks of caller ownership and unchanged Linear conversion behavior."""

import argparse
import ast
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import torch


def extract(path, scope):
    tree = ast.parse(path.read_text())
    function, = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "convert_exl3"]
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["convert_exl3"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("patched", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = torch.arange(128, dtype=torch.float).reshape(16, 8).half() / 8
    results = []

    class LinearFP16:
        swap_device = "cuda:0"

        def __init__(self):
            self.weight = source.clone()
            self.bias = torch.arange(8).half()

        def get_weight_tensor(self):
            return self.weight

        def get_bias_tensor(self):
            return self.bias

    for compact in (False, True):
        for return_weight in (False, True):
            variants = []
            for path in (args.original, args.patched):
                observed = {}

                def quantize(weight, h_data, quant_args, requested, progress, verbose, swap, save_reg=None):
                    observed.update(dtype=str(weight.dtype), h_data=h_data, quant_args=dict(quant_args),
                                    requested=requested, progress=progress, verbose=verbose, swap=swap, save_reg=save_reg)
                    original_ref = weakref.ref(weight)
                    # Distinct CPU storage models completion of the device upload.
                    staged = weight.float().clone()
                    del weight
                    gc.collect()
                    observed["caller_retains_source_after_upload"] = original_ref() is not None
                    assert torch.equal(staged, source.float())
                    return staged if requested else None, 0.125, {}

                def inner_factory(*args, **kwargs):
                    return SimpleNamespace(bias=args[-2].clone(), key=kwargs["key"])

                scope = dict(torch=torch, LinearFP16=LinearFP16, LinearEXL3=inner_factory, quantize_exl3=quantize)
                convert = extract(path, scope)
                linear = SimpleNamespace(inner=LinearFP16(), config=object(), in_features=16,
                                         out_features=8, out_dtype=torch.half, key="lm_head")
                output = convert(linear, {"fixture": True}, {"compact_cpu_buffers": compact},
                                 "fixture", return_weight, False, None, "cuda:0")
                if return_weight:
                    assert output[0] == 0.125 and torch.equal(output[1], source.float())
                else:
                    assert output == 0.125
                assert torch.equal(linear.inner.bias, torch.arange(8).half())
                assert linear.inner.key == "lm_head"
                variants.append(observed)
            assert variants[0].pop("caller_retains_source_after_upload") is True
            assert variants[1].pop("caller_retains_source_after_upload") is False
            assert variants[0] == variants[1]
            results.append(dict(compact=compact, return_weight=return_weight,
                                original_retains_source=True, patched_releases_source=True,
                                values_arguments_bias_and_return_exact=True))
    result = dict(device="cpu", cases=results, gpu_or_quantization_executed=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
