# Upstreams and attribution

EXL3-AMD extends an existing runtime and AMD port. The root [MIT license](../LICENSE) covers project additions; vendored code retains [its own notices](../vendor/rocm-exl3/LICENSE). Models and datasets have separate terms.

| Source | Contribution |
| --- | --- |
| [Turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) | EXL3 format, converter, model/cache/generator, base kernels and packed decoding |
| [CarouselAether / rocm_exl3](https://github.com/CarouselAether/rocm_exl3/tree/550dcfed786ad7bffa08b7a6b2a216fc474cbbb5) | Direct AMD upstream, pinned at `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`: RDNA/HIP kernels, dispatch, compatibility and build integration |
| [Cornell RelaxML / QTIP](https://github.com/Cornell-RelaxML/qtip), [paper](https://arxiv.org/abs/2406.11235) | Methodological foundation; the full QTIP implementation is not the vendored inference backend |
| [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | Inherited Gated DeltaNet chunk/prefill implementation |
| [PyTorch](https://github.com/pytorch/pytorch), [Triton](https://github.com/triton-lang/triton), [ROCm](https://github.com/ROCm) | Tensor, compiler and GPU dependencies |
| Hugging Face [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [Safetensors](https://github.com/huggingface/safetensors) | Tokenization, templates and tensor containers |
| [Daniel Lougen / DJLougen](https://huggingface.co/DJLougen), [GestaltLabs EXL3 release](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB) | Published EXL3 quant and accompanying model card that helped guide this project's target direction and quantization/runtime investigation |

## What EXL3-AMD adds to Aether's ROCm port

The direct starting point is CarouselAether's source at `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`, built on Turboderp's ExLlamaV3. The AMD port already provides packed EXL3 inference, MTP support, quantized cache kernels, model-loading CLI utilities and an [HTTP server](../vendor/rocm-exl3/rocm_tools/exl3_server/README.md). Those capabilities belong to the upstream implementation.

Our project-specific code and modifications are:

- **AMD/WSL execution:** [dispatch compatibility](../src/quantlab/methods/exl3/compat.py), eligible shared packed 2–9-row projection kernels for MTP verification, graph integration and decode fusions. These extend inherited EXL3 decoding and RDNA primitives.
- **Long-context execution:** target/draft Q8/Q4 cache integration and CLI flags, with a locally developed Q8 scheduling/reduction profile over inherited packed attention. Its exact activation guards and fallback behavior are documented in [LONG-CONTEXT.md](LONG-CONTEXT.md).
- **Conversion/build repairs:** K2 encoder shared-memory fixes, bounded conversion buffers, Qwen text/source mapping, native build compatibility and guarded reuse of matching build objects. The quantization method and base converter remain upstream work.
- **User-facing runtime:** [model-independent registration](../scripts/register_runtime.py), the [CLI/monitor](../scripts/launch_runtime.py), extension hash verification, GPU lease and portable resource limits.
- **HTTP and tool handling:** our [text HTTP adapter](../src/quantlab/server.py), [model-engine integration](../scripts/serve_exl3.py) and [tool-call parser](../src/quantlab/tool_calls.py). The adapter validates requests, serializes GPU work, handles complete Qwen XML/Hermes JSON calls and returns recoverable protocol errors. The upstream server informed disconnect watching and stream cleanup.
- **Measurement/verification:** independent packed-weight/operator checks, regression fixtures, occupied-context validation and [MTP timing correction](../scripts/exl3_timing.py). These establish scoped execution evidence, not BF16 fidelity or general performance superiority.

Small-M WMMA experiments reuse CarouselAether's RDNA primitives and Turboderp's decoding. Native attention experiments reuse upstream kernels and the driver bridge. Adaptive drafting exposes an inherited generator feature. Optional experiments are documented in [OPTIMIZATION.md](OPTIMIZATION.md); inclusion does not mean each is selected or faster.

This is a comparison with the pinned revision, not an inventory of the latest upstream. Aether's HTTP server exposes more sampling choices and endpoints. Our adapter offers a narrower text/tool API with the validated AMD/WSL integration. A claim that one runtime is faster would require a matched model, hardware, prompt and timing comparison.

## Daniel Lougen and GestaltLabs

Daniel Lougen's [EXL3 quant and accompanying model card](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB) helped guide this project's target direction: a compact 27B native EXL3 deployment retaining integrated MTP. The published quantization and documentation served as concrete technical references for our quantization/runtime investigation. This is credit for technical guidance from a released artifact and its documentation. Our AMD integration and separately converted model are distinct work; no GestaltLabs weights or helper implementation are bundled.

The [kernel directory](../kernels/exl3/README.md) retains historical patches/operator tools and applicable licenses. Private manifests, research handoffs and bundled calibration/reference text are excluded.

This is not an official release from the credited upstreams. OMP assisted development and is not an inference dependency.
