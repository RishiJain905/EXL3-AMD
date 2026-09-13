"""CPU control-flow/packing checks; codec calls are deterministic test doubles."""

import argparse
import ast
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    names = {"block_trace", "block_proxy_error_streamed", "ldlq_packed_columns"}
    nodes = [n for n in ast.parse(args.source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    calls = []

    def ldlq(weight, lower, qa):
        calls.append(tuple(weight.shape))
        assert weight.is_contiguous()
        q = weight.half()
        k, n = weight.shape
        encoded = q.view(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).contiguous().view(torch.int16)
        return q, encoded.view(k // 16, n // 16, 256)

    def pack(encoded, qa):
        return encoded[..., :16 * qa["K"]].contiguous()

    scope = dict(torch=torch, ldlq=ldlq, pack_trellis=pack, report_quant_memory=lambda *a: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.source), "exec"), scope)
    generator = torch.Generator().manual_seed(20260909)
    w = torch.randn(128, 8448, generator=generator) * torch.linspace(0.1, 10, 8448)
    h_base = torch.randn(128, 128, generator=generator)
    h = h_base @ h_base.T + torch.eye(128)
    lower = torch.eye(128)
    qa = {"K": 4, "compact_cpu_buffers": True, "devices": ["cpu"]}
    rng_before = torch.get_rng_state().clone()
    proxy, packed = scope["ldlq_packed_columns"](w, lower, h, qa)
    assert calls == [(128, 4096), (128, 4096), (128, 256)], calls
    q, encoded = ldlq(w, lower, qa)
    assert torch.equal(packed, pack(encoded, qa))
    expected = scope["block_proxy_error_streamed"](w, q, h, "cpu")
    relative = abs(proxy - expected) / max(abs(expected), 1e-20)
    assert relative <= 1e-5
    assert torch.equal(rng_before, torch.get_rng_state())
    rejected = 0
    for weight, H, config, block in (
        (w.half(), h, qa, 4096), (w[:, :-1], h, qa, 4096),
        (w, h, {"K": 4}, 4096), (w, h, qa, 127),
        (w, h[:-1], qa, 4096),
        (w, h, {**qa, "devices": ["cpu", "cpu"]}, 4096),
    ):
        try:
            scope["ldlq_packed_columns"](weight, lower, H, config, block)
        except AssertionError:
            rejected += 1
    assert rejected == 6
    result = dict(device="cpu", shape=list(w.shape), chunk_widths=[4096, 4096, 256],
                  test_double_packed_exact=True, proxy_relative_difference=relative,
                  rng_unchanged=True, invalid_arguments_rejected=rejected,
                  native_codec_or_gpu_executed=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
