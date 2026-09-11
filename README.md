# EXL3-AMD

A model-selectable EXL3 inference runtime for AMD GPUs, with a Python CLI and a local OpenAI-compatible HTTP server. Built on [Turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [CarouselAether's ROCm port](https://github.com/CarouselAether/rocm_exl3/tree/550dcfed786ad7bffa08b7a6b2a216fc474cbbb5).

**Status: experimental source release.** PowerShell launches inference through WSL; Linux can launch the backend directly. The validated GPU path is RDNA3/gfx1101 with a Qwen-family EXL3 model. Other models are selectable when supported by the vendored backend, but are not all validated. Models, native binaries and a turnkey installer are not bundled.

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
python run.py -m "MODEL_DIRECTORY" -c 4096 -n 256 -p "Write Python binary search." --execute
python run.py speed -m "MODEL_DIRECTORY" --cache-type q8 -c 4096 --execute
python run.py quality -m "MODEL_DIRECTORY" --cache-type q8 -c 4096 --execute

# Requires integrated MTP weights.
python run.py -m "MODEL_DIRECTORY" --cache-type q8 --spec-type draft-mtp --spec-draft-n-max 4 -c 4096 -n 256 -p "Explain parameterized SQL queries." --execute
```

Use `python3` on Linux. Sampling is greedy and thinking disabled. Speed mode uses warmed fixed-length continuations; quality mode is a small coding/SQL/security smoke suite. [Measurement and tuning](docs/OPTIMIZATION.md).

## Serve a harness

```powershell
python run.py serve -m "MODEL_DIRECTORY" --cache-type q8 --spec-type draft-mtp --spec-draft-n-max 4 -c 4096 --alias exl3 --port 8000 --execute
```

Omit MTP flags for models without MTP. Wait for `http://127.0.0.1:8000/health`, then use OpenAI Chat Completions with base URL `http://127.0.0.1:8000/v1`, model `exl3`, `temperature: 0`, `top_p: 1` and `n: 1`.

The server has no overall lifetime timeout. Explicit shutdown and resource/error safeguards remain active. `--request-timeout` applies to individual requests. Serving is loopback-only, with one active GPU request and four queued requests.

[HTTP API](docs/SERVING.md) · [Client-independent function tools](docs/TOOL-CALLING.md)

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
| `--execute` | Execute using enabled local permissions |

Context must fit model metadata and memory, including output and draft reserve. Allocation alone is not useful-context validation. GGUF names such as `q8_0`, separate draft-model files and llama.cpp offload flags are unsupported. See [KV cache](docs/KV-CACHE.md) and [long context](docs/LONG-CONTEXT.md).

Resource flags: `--gpu-memory-fraction` (0.90), `--max-gpu-memory-fraction` (0.95), `--max-host-memory-fraction` (0.50), `--min-free-ram-gib` (1) and `--min-free-disk-gib` (1). [Scope and limits](docs/DEPENDENCIES.md#resource-controls).

## Development and limitations

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

Some tests need inference or HTTP dependencies. Skips are not GPU validation. A clean build and a broad hardware/model matrix remain release work. No universal throughput, BF16 fidelity, stochastic sampling or unattended-service claim is made.

Run artifacts can contain prompts, outputs, token IDs and local paths. Keep them in ignored `artifacts/`; installation records belong in ignored `.runtime/` or local configs. Models, native binaries, developer jobs, private research history and calibration/reference corpora are excluded from Git.

## Credits and license

Thanks to **[Daniel Lougen / DJLougen](https://huggingface.co/DJLougen) and GestaltLabs** for the [Qwen3.8-27B-EXL3-11.5GB release](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB), a motivating example of practical EXL3 publishing. No weights from that release are bundled.

Project additions include CLI/HTTP integration, compatibility repairs and AMD runtime/kernel changes. [UPSTREAMS.md](docs/UPSTREAMS.md) identifies inherited implementations and authors. Project code is [MIT licensed](LICENSE); vendored code retains its [upstream notices](vendor/rocm-exl3/LICENSE). Model and dataset terms are separate. See [NOTICE](NOTICE).
