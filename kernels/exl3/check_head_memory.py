"""CPU-only parity checks against functions extracted from pinned Python source.

Requires the existing Torch environment. Does not import ExLlamaV3, initialize
an accelerator, or execute quantization. Pass original and patched quantize.py.
"""

import argparse
import ast
import json
from pathlib import Path

import torch


def extract(path, names):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    scope = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("patched", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    original = extract(args.original, ["block_trace"])["block_trace"]
    patched = extract(args.patched, ["block_trace", "block_proxy_error_streamed"])
    before_trace = next(n for n in ast.parse(args.original.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "block_trace")
    after_trace = next(n for n in ast.parse(args.patched.read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == "block_trace")
    assert ast.dump(before_trace) == ast.dump(after_trace)

    # Exhaust all finite half bit patterns. Decoder outputs are explicitly
    # half->float in the native kernel, so this covers their storage range.
    half = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    half = half[torch.isfinite(half)]
    wide = half.float()
    assert torch.equal(wide, wide.half().float())

    generator = torch.Generator().manual_seed(20260909)
    results = []
    for k, n, columns in ((65, 129, 32), (257, 2059, 1024), (1041, 37, 16)):
        w = torch.randn(k, n, generator=generator)
        q = (w + 0.1 * torch.randn(k, n, generator=generator)).half()
        a = torch.randn(k, min(k, 64), generator=generator)
        h = a @ a.T + torch.eye(k) * 0.1
        expected = original(w - q.float(), h) / max(original(w, h), 1e-8)
        actual = patched["block_proxy_error_streamed"](w, q, h, "cpu", columns)
        relative = abs(actual - expected) / max(abs(expected), 1e-12)
        assert relative <= 1e-5, (expected, actual, relative)
        # CPU half storage followed by explicit FP32 staging preserves the
        # error and LDLQ addmm operands/results bit for bit on this fixture.
        staged = torch.zeros_like(q, dtype=torch.float)
        staged.copy_(q)
        assert torch.equal(staged, q.float())
        lhs = torch.randn(k, 16, generator=generator)
        old_compensation = torch.zeros(16, n).addmm_(lhs.T, w - q.float())
        new_compensation = torch.zeros(16, n).addmm_(lhs.T, w - staged)
        assert torch.equal(old_compensation, new_compensation)
        results.append(dict(k=k, n=n, column_block_size=columns,
                            original_proxy=expected, streamed_proxy=actual,
                            relative_difference=relative, compensation_exact=True))

    zero = torch.zeros(17, 33)
    assert patched["block_proxy_error_streamed"](zero, zero.half(), torch.eye(17), "cpu", 8) == 0.0
    output = dict(device="cpu", torch=torch.__version__, finite_half_patterns=half.numel(),
                  half_storage_exact=True, zero_proxy_exact=True,
                  relative_metric_tolerance=1e-5, cases=results,
                  gpu_or_quantization_executed=False)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
