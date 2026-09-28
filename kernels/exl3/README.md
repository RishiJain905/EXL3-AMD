# EXL3 kernel development

Sources, patches and operator validators supporting [the vendored runtime](../../vendor/rocm-exl3/). The vendored tree is the active build source. Historical patch files are references and should not be blindly reapplied.

Developer tools include packed projection/small-M validation and conversion repairs. Some require explicit tensors, configurations or build outputs; they are not an installer or model download.

EXL3 decoding and base kernels originate with Turboderp/ExLlamaV3; RDNA/HIP primitives and the direct port originate with CarouselAether. Local changes reuse those components. See [UPSTREAMS.md](../../docs/UPSTREAMS.md), [LICENSE.upstream](LICENSE.upstream), and [rdna-smallm-LICENSE.txt](rdna-smallm-LICENSE.txt).

Use isolated matching extensions and known fixtures for operator checks. Keep outputs and private paths outside Git. CPU tests do not validate GPU kernels or full models. [Build instructions](../../docs/BUILD.md).

`check_smallm.py` independently decodes synthetic cb0/mul1 tensors on CPU and
checks native projections and graph replay. `check_wmma.hip` checks the actual
shared WMMA header and gfx12 fragment adapters with scalar CPU references.
The supervised `scripts/check_exl3_kernels.py` command runs small-M and prefill
checks with registered permissions, hash verification and GPU ownership.
See [GPU compatibility](../../docs/GPU-COMPATIBILITY.md) for commands and limits.

`check_highbit_smallm.py` covers the retained MiMo mul1 K5/K6 dot extension.
It also exercises every individual K5/K6 packed bit position and trellis wrap
boundary against the CPU oracle. See the [native kernel report](../../docs/NATIVE-KERNEL-PERFORMANCE.md)
for the aligned unpacker and projection-loop qualification.
`check_packed_mid.py` checks every row count from 10 through 64 against an
independent CPU reference, including output/scratch guards, non-default streams
and external graph replay with changed inputs. Select these suites with
`--checks highbit-smallm packed-mid`. See the [packed MLP report](../../docs/PACKED-MLP-PERFORMANCE.md)
for capability gates, tuning scope and validation status.

`check_packed_prefill.py` validates tiled packed projections from 65–4096 rows
against independent CPU weights, including tails and long reductions.
`check_mlp_pair.py` requires exact agreement with separate gate/up/SwiGLU
operations and covers malformed buffers, alignment, streams, and external
graph replay. Run them with `--checks packed-prefill mlp-pair` and a matching
verified build. See the [prefill/MLP report](../../docs/PREFILL-MLP-FUSION.md).

`check_head_tiled.py` covers wide FP16 and FP32 vocabulary heads against independently
decoded weights and exact inherited outputs. It also checks the compressed
layout adapter's shared-copy lifecycle, memory fallback, streams and graph
replay. `check_attention_schedule.py` checks format/geometry-specific attention
options with shuffled pages, output guards and changed-length graph replay.
Run `--checks head-tiled attention-schedule` with the matching verified build.
The [head/attention report](../../docs/HEAD-ATTENTION-PERFORMANCE.md) separates
operator checks from full-model and MTP evidence.

`check_gdn_recurrent.py` compares native GDN outputs and recurrent state with
an independent float64 NumPy recurrence. It covers saved history, native rewind
and suffix replay, grouped heads, generic/unsplit fallbacks, canaries,
non-default streams, changed-input/slot graph replay, and BF16 truncation.
`check_gdn_conv.py` independently checks native convolution outputs, complete
state windows, width fallbacks, bias/activation variants, graph replay and
native rewind followed by suffix replay. Run
`--checks gdn-recurrent gdn-conv` through the supervised validator. See the
[native kernel report](../../docs/NATIVE-KERNEL-PERFORMANCE.md) for qualification.
The compact recurrent implementation lives in the vendored `gdn.cu` and is
automatic for eligible multi-token ROCm calls. Single-token recurrence and
convolution retain their previous kernels; the independent convolution checker
is retained to catch regressions in future changes.

`check_hgemm.py` exposes `run_checks(torch, extension)` for the optional FP32-output
prefill GEMM. The caller must already hold the GPU lease and supply a verified,
loaded extension; the validator neither loads nor builds a binary. It checks
numerical references, dispatch fallbacks, output canaries, invalid arguments,
stream behavior and graph replay, including both sides of the 512-row WMMA
dispatch boundary. [Initial measurements](../../docs/GPU-PERFORMANCE.md) and
[full CLI follow-up](../../docs/RUNTIME-PERFORMANCE.md).

`check_kv_cache.py` and `check_asterkv.py` check packed cache storage and online
attention, independently of model task quality. With an existing registered
Linux/WSL installation, run the supervised validator from this checkout:

```bash
python scripts/check_exl3_kernels.py --checks kv-cache asterkv --output artifacts/kv-checks
```

The result directory must be new. Registration permissions, extension hashing,
the GPU lease and resource monitoring remain active. `asterkv_reference.py`
provides a separate NumPy quantization/calibration reference; its byte-sized
codes are unpacked reference data, not the runtime's five-bit storage.

`profile_asterkv.py` exposes `run_profile(torch, extension, ...)` to a caller
that already owns the GPU lease and has verified the extension. It compares a
bounded set of schedules on seeded synthetic caches and checks each output
against its codec's default. Optional capacity, occupancy, query lengths and
graph-replay timing support long-context diagnosis without loading a model.
Metadata records effective staging and decode-profile settings; these must
match serving when making a serving-related comparison. Microbenchmarks do
not establish end-to-end speed or task quality.
