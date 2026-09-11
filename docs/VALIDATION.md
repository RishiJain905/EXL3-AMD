# Release validation

Validated on 2026-09-11. This is an existing-environment source export smoke test, not a clean-install or broad compatibility certification.

## CPU checks

- Windows: 355 tests, successful, 14 skipped.
- Linux inference environment: 355 tests, successful, 10 skipped.
- CLI help works without model loading or installation metadata.

Coverage includes launcher boundaries, model-free registration, resource limits, telemetry selection, cache/tool protocols, timing, HTTP behavior and socket lifecycle. Platform/dependency skips remain explicit. New checks exercise different detected GPU/RAM totals, configurable stop thresholds, missing/ambiguous adapter handling, allocator forwarding, and persistent-server timeout behavior. OMP's default model assisted implementation; final review and acceptance used the actual exported files.

## GPU server smoke

The public source loaded a local 27B Qwen-family EXL3 model on gfx1101 through an existing compatible ROCm environment and a previously verified native extension. No weights or binary are distributed.

| Setting / observation | Result |
| --- | --- |
| Context / cache | 4096 / Q8 K and V |
| Integrated MTP | Maximum 6, adaptive confidence 0.6 |
| Attention profile | Default |
| Torch allocator fraction | 0.85 for this bounded smoke |
| Model loads | 1 |
| Generation requests completed | 14 of 14 |
| GPU engine failures | 0 |
| Readiness | 58.93 seconds |
| Entire acceptance including shutdown | 95.33 seconds |
| Shutdown | Explicit stop; clean shutdown confirmed |

Checks covered basic generation, four repeated raw completions, multi-turn chat, named/required function calls, SSE, tool-result turns, multiple/reordered results, deliberate truncation errors and subsequent recovery. The two deliberately invalid tool outputs produced the expected protocol errors while the engine stayed healthy. No model-generated commands were executed.

The worker recorded `timeout: null` for serving. Resource guards stayed active; the smoke's external supervisor imposed its own bounded acceptance deadline and stopped the server explicitly. This does not reintroduce a server lifetime cap.

## Export review

The public tree excludes models, compiled binaries, private installation/configuration and build records, raw logs, reference/calibration corpora, developer jobs and research handoffs. Public documentation links were checked. A file-content scan checked for local identifiers/paths and credential patterns. Source files and recorded local configuration in the original runtime were fingerprinted before and after export and matched.

Raw acceptance logs, responses, environment details and fingerprints remain private. They are not required to launch the source release.

## Remaining limits

No new native build, package installation, throughput optimization, full 98K/120K run or BF16 comparison was performed for this export. The smoke reused an existing compatible binary; it does not establish that a fresh full build reproduces it. Other GPU architectures, other models/templates, broad quality and prolonged serving need separate validation. Q4/MTP exact equivalence remains unresolved. No 60/80 tok/s or universal harness/model claim follows from this acceptance.
