"""Compare original and blocked tile sampling without importing the EXL3 runtime.

CPU is the default. An explicitly authorized GPU worker can select --device
cuda:0 to check reduction parity on that backend, without loading model weights.
"""

import argparse
import ast
from functools import lru_cache
import json
from pathlib import Path

import torch


def extract(path):
    tree = ast.parse(path.read_text())
    names = {"tensor_core_perm", "sample_scale_tiles"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    sampler = next(n for n in nodes if n.name == "sample_scale_tiles")
    assert isinstance(sampler.body[-1], ast.Return)
    # Expose intermediates only in this test copy; the production source stays unchanged.
    sampler.body[-1].value = ast.Tuple(
        elts=[sampler.body[-1].value] + [ast.Name(id=n, ctx=ast.Load()) for n in ("tile_ms", "hi", "lo")],
        ctx=ast.Load(),
    )
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    scope = {"torch": torch, "lru_cache": lru_cache}
    exec(compile(module, str(path), "exec"), scope)
    return scope["sample_scale_tiles"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("patched", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    original, patched = extract(args.original), extract(args.patched)
    generator = torch.Generator().manual_seed(20260909)
    results = []
    # Full output width exercises the production reduction stride with nine tile
    # rows: two full blocks and a final partial block. Other fixtures cover ties.
    for kind, k, n in (("random", 144, 248320), ("random", 272, 384),
                       ("zero", 144, 256), ("tied", 144, 384),
                       ("dynamic_range", 160, 1024)):
        weight = torch.randn(k, n, generator=generator)
        if kind == "zero":
            weight.zero_()
        elif kind == "tied":
            weight.fill_(1.0)
            weight[::16, ::16] = 2.0
        elif kind == "dynamic_range":
            weight *= torch.logspace(-4, 4, n).unsqueeze(0)
        weight = weight.to(device)
        rng_before = torch.get_rng_state().clone()
        accelerator_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        expected = original(weight)
        actual = patched(weight, tile_row_block_size=4)
        assert torch.equal(rng_before, torch.get_rng_state())
        if accelerator_before is not None:
            assert torch.equal(accelerator_before, torch.cuda.get_rng_state(device))
        energies_exact = torch.equal(expected[1], actual[1])
        max_relative = ((expected[1] - actual[1]).abs() / expected[1].abs().clamp_min(1e-30)).max().item()
        assert max_relative <= 1e-6, (kind, max_relative)
        assert torch.equal(expected[2], actual[2]), (kind, "high-energy tile indices changed")
        assert torch.equal(expected[3], actual[3]), (kind, "low-energy tile indices changed")
        assert torch.equal(expected[0], actual[0]), (kind, "sample tiles changed")
        results.append(dict(kind=kind, shape=[k, n], energies_exact=energies_exact,
                            maximum_relative_energy_difference=max_relative,
                            extreme_indices_exact=True, sampled_tiles_exact=True, rng_unchanged=True))
        del weight, expected, actual
    result = dict(device=str(device), torch=torch.__version__, cases=results,
                  quantization_executed=False, model_loaded=False,
                  interpretation="Sampling parity only; this does not prove packed-byte parity on untested tensors.")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
