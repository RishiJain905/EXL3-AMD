# Optional Qwen3.5 vision

`--mmproj on` loads the vision transformer and merger/projector already bundled
inside a supported dense Qwen3.5 EXL3 directory. `--mmproj off` is the default:
no vision model is constructed or loaded, and image requests are rejected.
Switch modes by stopping and restarting the process. Model files remain intact.

## CLI

```powershell
python run.py -m "MODEL_DIRECTORY" --mmproj on --decode-fusions off --cache-type f16 -c 4096 --image "picture.png" -n 128 -p "Describe the image."
```

`--image` is a local PNG/JPEG path and may be repeated up to four times. It
applies to generate mode. Generate uses the same image-aware engine as serve.
Vision currently supports generate and serve; benchmark modes keep their
existing text-only path. Neither mode downloads a model or projector.

For the separately qualified BF16 MTP package, add:

```text
--mtp 2 --mtp-dtype bf16 --verify-attention rowwise
```

That combination requires the verified K5/K6 extension and launch worksheet
described in [MIMO-MTP.md](MIMO-MTP.md). These flags do not add missing MTP
weights or native capabilities to another package or binary.

## HTTP

```powershell
python run.py serve -m "MODEL_DIRECTORY" --mmproj on --decode-fusions off --cache-type f16 -c 4096 -b 256 --alias exl3 --port 8000
```

Send user content in `/v1/chat/completions`:

```json
{
  "model": "exl3",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64_IMAGE_BYTES"}},
    {"type": "text", "text": "Describe the image."}
  ]}],
  "max_tokens": 128,
  "stream": false
}
```

PNG and JPEG base64 data URLs are supported. Optional image `detail` must be
`auto`; control processing resolution at startup instead. HTTP requests cannot
fetch remote URLs or read local paths. Images are accepted in user messages,
including conversation history. Streaming uses the existing SSE format.
`/health` exposes the `vision` status and whether the component is loaded.
Raw `/v1/completions` remains text-only.

## Memory and limits

| Setting | Limit |
| --- | --- |
| Processed resolution | `--image-max-pixels 262144` by default; aspect ratio retained and dimensions aligned to patches |
| Configurable ceiling | Multiple of 1024 in [65536, 1048576], within source processor limits |
| Source image | At most 8 MiB compressed and 16 megapixels; single-frame PNG/JPEG |
| Images per request | At most four, subject to context space |
| HTTP body | 12 MiB with vision enabled; 1 MiB with vision off |
| Context | Expanded image tokens plus text, requested output and draft reserve must fit |
| Current runtime envelope | Dense Qwen3.5, no deepstack layers, F16 KV, decode fusions off |

The processed-pixel cap can downsize a large image. With patch16/merge2, a
256-square image contributes 64 image tokens; 512-square contributes 256;
1024-square contributes 1024. Start/end markers and surrounding text use
additional tokens. Do not assume the source processor's much larger ceiling
fits a consumer GPU.

The vision transformer remains resident between requests. Temporary image
tensors may also increase the allocator's reserved memory; restart with
`--mmproj off` to obtain the text-only residency. This is not dynamic unloading
and does not preserve an old process's image-derived KV state. `off` does not
run the vision transformer on the CPU. Host image decoding and embedding
transfer are part of the inherited input path.

The original BF16 vision files are preserved; vision inference uses the
inherited mixed FP16/FP32 implementation. The MTP BF16 option applies to the
draft weights/projections, not to every vision operator. GPU embedding/draft,
native attention, draft graphs/shortlists, projection caching and prefix reuse
are excluded from this first vision integration envelope.

## Implementation and validation

The implementation reuses the vendored Qwen3.5/Qwen3-VL component, smart resize,
patchification, MRoPE and MMEmbedding support. It uses the model's own processor
metadata and chat template. A full image alias expands into one pair of vision
boundaries, avoiding duplicate markers. Context admission happens before vision
GPU work; embedding lengths and finite values are checked before generation.

EXIF orientation is applied and transparent images are composited onto white.
Weights, metadata and tokenizers are never edited to enable vision. The native
binary is unchanged by this feature.

The MiMo 5.0/H6 mul1 package with BF16 MTP depth 2 passed the bounded RX 7800 XT
integration checks at context 4096, F16 KV, prefill 256, rowwise verification
and decode fusions off. Highest sampled whole-adapter peaks across Step 5:

| Mode | Peak adapter VRAM |
| --- | ---: |
| Vision off, text and BF16 MTP | 7.265 GiB |
| Vision on, processing cap 262144 pixels | 7.976 GiB |
| Vision on, processing cap 1048576 pixels | 10.174 GiB |

These include desktop/driver memory and actual image requests where enabled.
They are observations, not maximum-memory guarantees. The repeated single-image
protocol measured 7.935 and 10.111 GiB respectively; the image was processed at
512×512 or 1024×1024, with one warmup and three 128-token continuations. Live
Torch allocation increased by 0.882 GiB when loading vision. Image workspaces
also increased reserved memory, and a fresh off restart restored text residency.

Exact MTP-off/on generated IDs and stop conditions matched for selection
(259 IDs), confirmation with thinking off (325) and thinking on (352). The
public Windows CLI preserved all 45 input and 42 output IDs when toggling
vision residency on a text request. Loopback image JSON/SSE, disabled-image
rejection and post-image text requests passed. CPU suites passed 643 tests
on Windows (46 skips) and 679 on Linux (10 Windows-only skips).

Detailed protocols, failures, source hashes and measurements are tracked in the
[Step 5 research report](https://github.com/RishiJain905/QuantizationResearch/blob/codex/mimo-vision-20260925/docs/mimo-exl3-step5.md).
Authored image checks establish a bounded integration result; they do not
establish general vision quality, long-context behavior or Q6/Q8 equivalence.
