# 27B and MiMo launch commands and flag reference

Profiles recorded September 27, 2026, for RX 7800 XT / gfx1101. These use
the latest qualified native optimizations automatically. They are the current
validated profiles; a complete sweep of every flag combination on the newest
binary has not been performed. See [measured results](NATIVE-KERNEL-PERFORMANCE.md).

## Launch commands

Run these in PowerShell, one model at a time. Replace `PATH_TO_EXL3_AMD`,
`QWENSEEK_MODEL_DIRECTORY` and `MIMO_MODEL_DIRECTORY` with your checkout and
complete local EXL3 model directories. A compatible native extension must
already be built and registered as described in [BUILD.md](BUILD.md). Pulling
source alone does not replace an existing registered binary.

```powershell
cd "PATH_TO_EXL3_AMD"
```

### Qwenseek 27B

Q6 KV cache, integrated MTP depth two, and GDN decode fusions:

```powershell
python run.py serve `
  -m "QWENSEEK_MODEL_DIRECTORY" `
  -c 16384 `
  -b 1024 `
  --mtp 2 `
  --cache-type q6 `
  --decode-fusions gdn `
  --smallm-kernel dot `
  --prefill-gemm auto `
  --attention-profile auto `
  --prefix-cache off `
  --mmproj off `
  --reasoning auto `
  --temperature 0 `
  --alias qwenseek-27b `
  --port 8000
```

### MiMo 9B

BF16 MTP depth two, rowwise verification, and F16 KV cache:

```powershell
python run.py serve `
  -m "MIMO_MODEL_DIRECTORY" `
  -c 16384 `
  -b 1024 `
  --mtp 2 `
  --mtp-dtype bf16 `
  --verify-attention rowwise `
  --cache-type f16 `
  --decode-fusions off `
  --smallm-kernel dot `
  --prefill-gemm auto `
  --attention-profile auto `
  --prefix-cache off `
  --mmproj off `
  --reasoning auto `
  --temperature 0 `
  --alias mimo-9b `
  --port 8000
```

Both expose `http://127.0.0.1:8000/v1`. Use the matching alias as the API model
name. Wait for readiness before sending requests. Several defaults are written
explicitly so the configuration is clear.

MiMo's `--decode-fusions off` preserves its validated BF16 MTP configuration.
The new recurrent kernel still runs for eligible multi-token calls. Details
are in [MIMO-MTP.md](MIMO-MTP.md).

The examples allocate a 16,384-token context, including prompt, output and MTP
reserve. The latest kernel qualification used capacity 16,896 to accommodate
an occupied 16K prompt and its continuation. Choose context capacity for the
actual workload; a larger capacity is not itself a speed optimization.

Serving has no default request deadline or overall lifetime timeout. Generation
uses the remaining context unless the API request explicitly supplies
`max_tokens` or `max_completion_tokens`; omit them or send `null` for no client
output cap. `-n` applies to generate mode. Remove an old `--request-timeout 900`
argument or use `--request-timeout 0` to disable that explicit deadline.
See [SERVING.md](SERVING.md).

### Jinja and GPU memory allowance

Jinja chat templating is already active. `--jinja` is accepted for compatibility
and changes no behavior or performance. There is no `--ninja` runtime flag.

The commands omit `--gpu-memory-fraction`, using its normal `0.90` default.
The earlier `0.88` override reproduced the benchmark's allocation budget; no
speed benefit from choosing `0.88` was established. On a 16 GiB GPU these
fractions allow approximately 14.4 GiB and 14.1 GiB, respectively, through the
Torch allocator. They do not cap compute utilization or memory bandwidth.

The remaining headroom accommodates the display and allocations outside that
allocator. The separate `--max-gpu-memory-fraction` monitor defaults to `0.95`;
it is a sampled adapter-usage threshold, not the Torch allocation allowance.
See [resource controls](DEPENDENCIES.md#resource-controls).

### Coding sampling versus benchmark settings

The commands use greedy sampling (`--temperature 0`) to match the performance
and output-comparison protocol. That setting has not been established as the
best coding-quality choice. The evaluated model packages supply these sampled
generation defaults, which are starting points for everyday coding:

| Model package | Server sampling flags |
| --- | --- |
| MiMo 9B | `--temperature 0.6 --top-p 0.95 --top-k 20` |
| Qwenseek 27B | `--temperature 1.0 --top-p 0.95 --top-k 20` |

Replace the temperature argument and add the other two flags when using these
settings. The runtime uses its explicit CLI/request sampling controls; merely
having `generation_config.json` beside the weights does not apply these defaults.
A coding-quality temperature sweep has not been performed, and sampled
throughput need not match the greedy measurements.

## OpenCode MiMo context profiles

These example OpenCode profiles use an OpenAI-compatible provider with text
input/output, tool calling and interleaved `reasoning_content`. Configure the
model IDs to match the names below, an 8,192-token output limit, and the stated
context limits and endpoints. Client settings remain local and are not bundled
with the runtime repository.

| OpenCode model | Total context capacity | Endpoint | Launch cache / verification |
| --- | ---: | --- | --- |
| `Mimo9B-131k` | 131,072 | `http://127.0.0.1:8093/v1` | F16 / rowwise |
| `Mimo9B-256k` | 262,144 | `http://127.0.0.1:8094/v1` | Q8 / default |

OpenCode's context limit is client bookkeeping; selecting an entry does not
start an inference server. From the repository directory, start the matching
command below, then select that model in OpenCode. Run one server at a time.
The alias must match the configured model ID.

For `Mimo9B-131k`:

```powershell
python run.py serve `
  -m "MIMO_MODEL_DIRECTORY" `
  -c 131072 -b 1024 `
  --mtp 2 --mtp-dtype bf16 `
  --cache-type f16 --verify-attention rowwise `
  --decode-fusions off `
  --smallm-kernel dot --prefill-gemm auto --attention-profile auto `
  --prefix-cache off --mmproj off `
  --reasoning auto --temperature 0 `
  --alias Mimo9B-131k --port 8093
```

For `Mimo9B-256k`:

```powershell
python run.py serve `
  -m "MIMO_MODEL_DIRECTORY" `
  -c 262144 -b 1024 `
  --mtp 2 --mtp-dtype bf16 `
  --cache-type q8 --verify-attention default `
  --decode-fusions off `
  --smallm-kernel dot --prefill-gemm auto --attention-profile auto `
  --prefix-cache off --mmproj off `
  --reasoning auto --temperature 0 `
  --alias Mimo9B-256k --port 8094
```

These larger-context launch profiles have not been load-tested or qualified
with fully occupied prompts. Both capacities are within the model metadata's
262,144-token maximum; that alone does not establish long-context quality or
performance. Input, output and MTP reserve must fit the total capacity.

The 256K command changes the earlier F16/rowwise profile deliberately. Its
target and draft F16 KV tensors alone would occupy 9 GiB; scaling the latest
measured allocation gives an estimated 15.32 GiB peak before additional
long-context overhead, exceeding the default 14.4 GiB Torch allowance on a
16 GiB card. Q8 reduces the cache allocation, but requires default verification
because rowwise verification currently accepts only F16 KV. This Q8 combination
does not inherit the earlier F16/rowwise output-equivalence qualification.
Normal resource guards remain active; no larger memory allowance is forced.

## Useful performance experiments

Change one setting at a time, restart the server, and compare identical model
bytes, prompt tokens, output budgets and sampling. Record prefill, decode and
total request time separately. Model loading from disk is separate from those
inference measurements.

| Flag | Available values | What to try or expect |
| --- | --- | --- |
| `--mtp` | Integers `0`-`8` | Start with `1`, `2`, `4`. Higher depth can help when drafts are accepted, but can also slow decoding. `0` disables MTP. |
| `--draft-confidence` | Greater than `0`, less than `1` | Try `0.6` with positive MTP depth for adaptive drafting. Omit for fixed depth. Conflicts with GPU drafting and shortlists. |
| `-b` / `--prefill-chunk` | `256`-`8192`, multiples of `256` | Compare `512`, `1024`, `2048`. Larger chunks use more temporary VRAM. `1024` is the measured balanced choice. |
| `-c` / `--ctx-size` | At least `1024`, multiples of `256` | Examples: `8192`, `16384`, `32768`. Must fit model limits and available memory. Larger capacity does not inherently improve speed. |
| `--cache-type` | `f16`, `q8`, `q6`, `q5`, `q4`, experimental `aster5` | Trades cache memory, precision and performance. Keep MiMo's rowwise profile on `f16`. |
| `-ctk`, `-ctv` | Same cache choices | Set K/V precision separately. Both must be quantized or both `f16`; `aster5` requires both sides. |
| `--prefix-cache` | `off`, `on` | `on` can greatly reduce repeated-prefix prefill in ongoing conversations. Earlier 30K testing showed fresh-versus-cached output differences, so it remains optional. |
| `--gpu-memory-fraction` | Greater than `0`, less than `1`; default `0.90` | Changes PyTorch's VRAM allowance. It does not limit GPU compute or bandwidth. |

The prefix-cache limitation and measured reuse gains are documented in
[RUNTIME-PERFORMANCE.md](RUNTIME-PERFORMANCE.md#prefix-reuse-and-validation-limits).

## Kernel and MTP controls

These alternatives are not established improvements over the launch commands
above. Automatic policies already select the promoted implementations.

| Flag | Available values | Notes |
| --- | --- | --- |
| `--decode-fusions` | `off`, `gdn`, `gdn-mlp` | Keep `gdn` for the current 27B profile and `off` for MiMo BF16 MTP. |
| `--mtp-dtype` | `fp16`, `bf16` | BF16 requires a compatible dense BF16 draft package. The 27B's quantized draft cannot simply be switched to BF16. |
| `--verify-attention` | `default`, `rowwise` | Use `default` with Q8/Q6/Q5/Q4. Rowwise requires F16 KV and disabled decode fusions. |
| `--smallm-kernel` | `dot`, `wmma`, `wmma-register` | `dot` is the qualified choice here; alternatives need compatible native support. |
| `--prefill-gemm` | `auto`, `blas`, `wmma` | Keep `auto`; forced choices are useful for comparisons. |
| `--attention-profile` | `auto`, `default`, `long` | `auto` uses the measured schedules. `default` selects inherited scheduling; `long` is the legacy long-context policy. |
| `--warps`, `--smallm-mlp-warps` | `4`, `8`, `16` | Scheduling overrides. Leave unset for the selected policy. |
| `--head-warps` | `1`, `4`, `8`, `16` | Vocabulary-head scheduling override; leave unset normally. |
| `--cache-mtp` | `off`, `fc`, `attention`, `mlp`, `all` | Caches reconstructed draft projections using extra VRAM. Requires positive MTP and `off`/`gdn` fusions. Incompatible with MiMo's BF16 MTP setting. |
| `--gpu-embedding`, `--gpu-draft`, `--gpu-draft-metadata` | Presence enables each | Experimental GPU bookkeeping. Drafting requires GPU embedding and MTP; metadata requires GPU drafting. Extra VRAM needed. |
| `--no-packed-mid`, `--no-packed-prefill`, `--no-mlp-pair` | Presence disables each | Diagnostic comparisons against automatically enabled optimizations. |

To test `--mtp 0` on MiMo, also remove `--mtp-dtype bf16` or set it to `fp16`.
The qualified BF16/rowwise profile otherwise requires positive MTP depth,
F16 KV and `--decode-fusions off`. BF16 MTP also excludes native attention,
the draft-step graph and reconstructed MTP projection caching.

## Other serving options

| Flag | Options / purpose |
| --- | --- |
| `--reasoning` | `auto`, `on`, `off`. Disabling reasoning can shorten answers; it changes model behavior. |
| `--temperature` | `0`-`2`; `0` is greedy, matching the performance comparisons. |
| `--top-p`, `--top-k`, `--min-p` | Sampling controls: respectively `(0,1]`, `0`-`1000`, `[0,1)`. |
| `--seed` | Fixes the sampling seed; integer in `[0, 2^63)`. |
| `--mmproj` | `off`, `on`; enables compatible bundled vision support with its additional memory requirements. See [VISION.md](VISION.md). |
| `--jinja` | Accepted, but templating already uses Jinja. |
| `--chat-template-file` | Overrides the model's template with a local file. |
| `--request-timeout` | `0` disables (default); any positive integer sets seconds per request, including queue time. |
| `--alias`, `--port` | API model name and listening port. |

Requests can override the supported sampling and reasoning defaults; keep
those choices fixed when comparing performance. The full command inventory,
including additional experimental and diagnostic options, is available with:

```powershell
python run.py --help
```
