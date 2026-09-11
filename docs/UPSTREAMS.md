# Upstreams and attribution

EXL3-AMD extends an existing runtime and AMD port. The root [MIT license](../LICENSE) covers project additions; vendored code retains [its own notices](../vendor/rocm-exl3/LICENSE). Models and datasets have separate terms.

| Source | Contribution |
| --- | --- |
| [Turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 format, converter, model/cache/generator, base kernels and packed decoding |
| [CarouselAether / rocm_exl3](https://github.com/CarouselAether/rocm_exl3/tree/550dcfed786ad7bffa08b7a6b2a216fc474cbbb5) | Direct AMD upstream, pinned at `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`: RDNA/HIP kernels, dispatch, compatibility and build integration |
| [Cornell RelaxML / QTIP](https://github.com/Cornell-RelaxML/qtip), [paper](https://arxiv.org/abs/2406.11235) | Methodological foundation; the full QTIP implementation is not the vendored inference backend |
| [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | Inherited Gated DeltaNet chunk/prefill implementation |
| [PyTorch](https://github.com/pytorch/pytorch), [Triton](https://github.com/triton-lang/triton), [ROCm](https://github.com/ROCm) | Tensor, compiler and GPU dependencies |
| Hugging Face Transformers, Tokenizers, Safetensors | Tokenization, templates and tensor containers |
| [Daniel Lougen / DJLougen](https://huggingface.co/DJLougen), [GestaltLabs EXL3 release](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB) | Motivating model publication; no weights or helper implementation imported from that release |

## Modifications in this project

The vendored tree includes build compatibility and guarded object reuse, encoder LDS repairs, memory-bounded conversion, Qwen text adaptation, noncooperative WSL dispatch, decode fusions, shared packed small-M kernels, graph/cache state repairs and quantized-cache attention integration.

Small-M WMMA paths reuse CarouselAether's RDNA primitives and Turboderp's decoding. Native attention reuses upstream kernels and the driver bridge. Adaptive drafting exposes an inherited generator feature. These are modifications of existing components, not claims to have invented EXL3, WMMA, MTP or graph capture.

Project code in `src/quantlab/` and `scripts/` adds the launcher, monitoring, compatibility integration, measurements and HTTP adapter. The inherited [upstream server](../vendor/rocm-exl3/rocm_tools/exl3_server/README.md) informed disconnect watching and stream cleanup; this adapter exposes a smaller API.

[kernels/exl3](../kernels/exl3/README.md) retains historical patches/operator tools and applicable licenses. Private manifests, research handoffs and bundled calibration/reference text are excluded.

This is not an official release from the credited upstreams. OMP assisted development and is not an inference dependency.
