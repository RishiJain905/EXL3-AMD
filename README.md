# EXL3-AMD

A model-selectable EXL3 inference runtime for AMD GPUs, with a Python CLI and a local OpenAI-compatible HTTP server. Built on [Turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [CarouselAether's ROCm port](https://github.com/CarouselAether/rocm_exl3/tree/550dcfed786ad7bffa08b7a6b2a216fc474cbbb5).

**Status: experimental source release.** PowerShell launches inference through WSL; Linux can launch the backend directly. The validated GPU path is RDNA3/gfx1101 with a Qwen-family EXL3 model. RDNA4/gfx1200/gfx1201 support is experimental and awaits hardware validation. Fused small-batch projections support the default and mul1 codebooks within the documented [compatibility envelope](docs/GPU-COMPATIBILITY.md). Other models are selectable when supported by the vendored backend, but are not all validated. Models, native binaries and a turnkey installer are not bundled.

| Component | Language / packages |
| --- | --- |
| CLI, installation record, monitoring | Python 3.11+, standard library, TOML |
| Model execution and tokenization | Python, PyTorch ROCm, Transformers, Tokenizers, Safetensors |
| GPU kernels | C++/HIP, ROCm, Triton, Flash Linear Attention |
| HTTP API and streaming | FastAPI, Starlette, Uvicorn |

[Build and setup](docs/BUILD.md) · [Dependencies](docs/DEPENDENCIES.md) · [EXL3 explained](docs/EXL3.md) · [Upstreams](docs/UPSTREAMS.md) · [Validation](docs/VALIDATION.md)

## Features

- Local EXL3 model directories selected with `-m`.
- Generation, speed/quality checks, occupied-context measurements and persistent serving.
- FP16, integer Q8 and integer Q4 attention KV caches.
- Integrated MTP drafting when the model includes compatible weights.
- OpenAI Chat Completions tools for recognized Qwen XML and Hermes JSON templates.
- Configurable resource limits, GPU lease, extension hash checks, offline loading and private artifacts.

EXL3 stores packed low-bit weights inside Safetensors. It needs a compatible loader and decode kernels; it cannot simply be handed to this project's llama.cpp/GGUF path. [Format and architecture](docs/EXL3.md).

## What we built on the upstream foundation

EXL3-AMD uses a modified copy of CarouselAether's ROCm port, pinned at `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`. Our contribution combines new project code with targeted changes to that inherited backend. The EXL3 format, base model execution, AMD port and integrated MTP implementation come from the upstream authors.

| Area | Upstream foundation | Our additions and modifications |
| --- | --- | --- |
| AMD execution and MTP verification | Turboderp's packed-weight decoding and CarouselAether's HIP/RDNA kernels and dispatch | WSL dispatch repairs, shared packed projection kernels for eligible 2–9 token rows, graph integration and decode fusions. The small-row kernels reuse decoded weight fragments during MTP verification. |
| KV cache and long context | Inherited packed Q8/Q4 storage, rotation and attention kernels | CLI/server cache selection for target and draft, plus an opt-in Q8 attention scheduling/reduction profile with explicit GPU, shape and context guards. |
| Conversion and native builds | ExLlamaV3's converter and the ROCm build system | K2 encoder shared-memory repairs, memory-bounded conversion buffers, Qwen text/source adaptation, build compatibility fixes and guarded object reuse. |
| CLI and installation | Upstream model loading, tokenization and generation | The `run.py` interface with familiar `-m`, `-c` and MTP flags; model-independent installation registration; extension verification, GPU ownership, configurable resource monitoring and private run artifacts. |
| Serving and function tools | An [HTTP server already exists upstream](vendor/rocm-exl3/rocm_tools/exl3_server/README.md); its disconnect/stream-cleanup design informed this work | Our text-only HTTP adapter, serialized request lifecycle, Qwen XML/Hermes JSON tool parsing, structured responses, validation and recoverable malformed-tool errors. Persistent serving has no overall lifetime cap. |
| Measurements and verification | Upstream operators, model/runtime interfaces and evaluation utilities | Independent packed-weight/operator checks, model and HTTP regression fixtures, occupied-context checks, and corrected MTP throughput accounting that groups tokens by emitting GPU iteration. |

The implementation is in [scripts](scripts/), [src/quantlab](src/quantlab/), [kernel work](kernels/exl3/README.md) and the modified [vendored backend](vendor/rocm-exl3/). [UPSTREAMS.md](docs/UPSTREAMS.md) gives component-level credit and source references.

This comparison is against the pinned upstream revision. Aether's server exposes broader sampling and endpoint options; our adapter concentrates on the documented text/tool contract and tested AMD/WSL path. These additions do not establish a general speed advantage over Aether's runtime or compatibility with every EXL3 model. See [validation scope](docs/VALIDATION.md) and [measurement guidance](docs/OPTIMIZATION.md).

## Install once

Provision a compatible ROCm environment and build the extension using [BUILD.md](docs/BUILD.md). Fill in the installation worksheet with your interpreter, SDK, GPU target and extension hash:

```powershell
Copy-Item configs/local.example.toml configs/local.toml
# Edit the local worksheet, including execution permissions.
python scripts/register_runtime.py configs/local.toml
```

Registration writes ignored `.runtime/installation.toml`. It installs nothing, selects no model, and refuses to overwrite an existing registration. Normal commands need no `--config`.

## Generate and benchmark

From the repository root, replace `MODEL_DIRECTORY` with a complete local EXL3 folder. Relative paths work; `-m` never downloads a model.

```powershell
python run.py -m "MODEL_DIRECTORY" -c 4096 -n 256 -p "Write Python binary search."
python run.py speed -m "MODEL_DIRECTORY" --cache-type q8 -c 4096
python run.py quality -m "MODEL_DIRECTORY" --cache-type q8 -c 4096

# Requires integrated MTP weights.
python run.py -m "MODEL_DIRECTORY" --cache-type q8 --spec-type draft-mtp --spec-draft-n-max 4 -c 4096 -n 256 -p "Explain parameterized SQL queries."
```

Use `python3` on Linux. Commands run once the registered installation allows local inference and backend probes. Benchmark modes use greedy sampling; serve defaults to greedy with per-request sampling overrides. Reasoning follows the model template by default. Speed mode uses warmed fixed-length continuations; quality mode is a small coding/SQL/security smoke suite. [Measurement and tuning](docs/OPTIMIZATION.md).

## Serve a harness

```powershell
python run.py serve -m "MODEL_DIRECTORY" --cache-type q8 --spec-type draft-mtp --spec-draft-n-max 4 -c 4096 --alias exl3 --port 8000
```

Omit MTP flags for models without MTP. Wait for `http://127.0.0.1:8000/health`, then use OpenAI Chat Completions with base URL `http://127.0.0.1:8000/v1`, model `exl3` and `n: 1`. Greedy decoding is the default; sampling, reasoning, and prefill-chunk controls are documented under [HTTP API](docs/SERVING.md).

The server has no overall lifetime timeout. Explicit shutdown and resource/error safeguards remain active. `--request-timeout` applies to individual requests. Serving is loopback-only, with one active GPU request and four queued requests.

[HTTP API](docs/SERVING.md) · [Client-independent function tools](docs/TOOL-CALLING.md)

[llama.cpp flag mapping and limits](docs/LLAMA-CPP-COMPATIBILITY.md).
[Storage, chunk tuning, and prefix-reuse measurements](docs/PREFILL-PERFORMANCE.md).
[Native prefill GPU measurements](docs/GPU-PERFORMANCE.md).
[Full CLI decode and prefill follow-up](docs/RUNTIME-PERFORMANCE.md).
[Mul1 model, speed and occupied 120K validation](docs/MUL1-VALIDATION.md).

## Main flags

`python run.py --help` lists all options. Modes: `generate` (default), `serve`, `speed`, `quality`, `context`, `context-speed`, `profile`.

| Flag | Meaning |
| --- | --- |
| `-m / --model` | EXL3 directory containing config, tokenizer and shards |
| `-c / --ctx-size` | Total context; default 4096, minimum 1024, multiple of 256 |
| `-n / --n-predict` | Generate-mode output limit, 1–8192 |
| `-p / --prompt`, `-f / --file` | Prompt text or UTF-8 file |
| `--cache-type f16\|q8\|q4` | Both KV precisions; default f16 |
| `-ctk`, `-ctv` | Separate K/V precision; both quantized or both f16 |
| `--spec-type draft-mtp` | Integrated MTP; drafting is off by default |
| `--spec-draft-n-max` | Draft depth 0–8; draft-mtp alone selects 2 |
| `--draft-confidence` | Optional adaptive confidence, strictly between 0 and 1 |
| `--attention-profile default\|long` | Default scheduling or guarded Q8 long-context optimization |
| `--output` | New private artifact directory; existing directories refused |
| `--reasoning on\|off\|auto` | Serve thinking-template mode; default auto respects the template |
| `--temperature`, `--top-p`, `--top-k`, `--min-p`, penalties, `--seed` | Serve sampling defaults with per-request overrides; default greedy |
| `-b / --prefill-chunk` | Prompt tokens per prefill step, 256–8192 in multiples of 256; default 1024 for serve, 256 otherwise |
| `--prefix-cache on\|off` | Serve: reuse matching KV/recurrent checkpoints; default off, up to 1 GiB host checkpoint memory |
| `-ncmoe / --n-cpu-moe`, `--cpu-moe all` | Keep first-N (or all) MoE-layer experts on CPU; MoE models only, not validated on GPU |

Context must fit model metadata and memory, including output, reasoning, and draft reserve. Allocation alone is not useful-context validation. GGUF names such as `q8_0`, separate draft-model files, and most llama.cpp knobs (`ngl`, `mmap`, RoPE overrides) are unsupported; only the mapped `-b` and `-ncmoe` analogues exist. See [KV cache](docs/KV-CACHE.md) and [long context](docs/LONG-CONTEXT.md).

Resource flags: `--gpu-memory-fraction` (0.90), `--max-gpu-memory-fraction` (0.95), `--max-host-memory-fraction` (0.50), `--min-free-ram-gib` (1) and `--min-free-disk-gib` (1). [Scope and limits](docs/DEPENDENCIES.md#resource-controls).

## Development and limitations

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

Some tests need inference or HTTP dependencies. Skips are not GPU validation. A clean build and a broad hardware/model matrix remain release work. No universal throughput, BF16 fidelity, broad sampling quality or unattended-service claim is made.

Run artifacts can contain prompts, outputs, token IDs and local paths. Keep them in ignored `artifacts/`; installation records belong in ignored `.runtime/` or local configs. Models, native binaries, developer jobs, private research history and calibration/reference corpora are excluded from Git.

## Upstream contributors and license

| Contributor / project | Work this runtime builds on |
| --- | --- |
| [Turboderp and ExLlamaV3 contributors](https://github.com/turboderp-org/exllamav3) | EXL3 format, converter, model/cache/generator, base kernels and packed-weight decoding. |
| [CarouselAether / rocm_exl3](https://github.com/CarouselAether/rocm_exl3/tree/550dcfed786ad7bffa08b7a6b2a216fc474cbbb5) | The direct AMD/ROCm upstream: HIP/RDNA kernels, dispatch, compatibility, build integration and the inherited HTTP server. |
| [Cornell RelaxML / QTIP](https://github.com/Cornell-RelaxML/qtip), [paper](https://arxiv.org/abs/2406.11235) | Quantization research underlying EXL3. |
| [Flash Linear Attention contributors](https://github.com/fla-org/flash-linear-attention) | Gated DeltaNet chunk/prefill operations used by the inherited model path. |
| [PyTorch](https://github.com/pytorch/pytorch), [Triton](https://github.com/triton-lang/triton), [AMD ROCm](https://github.com/ROCm) contributors | Tensor execution, GPU compilation and the ROCm software stack. |
| Hugging Face [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), [Safetensors](https://github.com/huggingface/safetensors) contributors | Model tokenization, chat templates and tensor containers. |
| [Daniel Lougen / DJLougen](https://huggingface.co/DJLougen) and [GestaltLabs](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB) | Published EXL3 quantization and model-card guidance that helped shape this project's target direction and quantization/runtime investigation. |

**Daniel Lougen's EXL3 quant and accompanying [Qwen3.8-27B-EXL3-11.5GB model card](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB) were concrete technical references for this work.** They helped guide our target direction toward a compact 27B native EXL3 deployment with integrated MTP and informed our quantization/runtime investigation. This credits the published quantization and documentation as technical guidance; our AMD integration and separately converted model are distinct work. No GestaltLabs weights or helper implementation are bundled here.

Project code is [MIT licensed](LICENSE); vendored code retains its [upstream notices](vendor/rocm-exl3/LICENSE). Model and dataset terms are separate. See [NOTICE](NOTICE) and [UPSTREAMS.md](docs/UPSTREAMS.md).
