# Multimodal Benchmarking

Examples for benchmarking VLMs and audio-capable models with synthetic image,
video, and audio payloads.

## Files

- **`image.yaml`** — Visual-QA-style workload: text prompt with N images per
  request. Typical first config to try for an image-capable VLM.
- **`video_mp4.yaml`** — Video benchmark using the `mp4` wire format
  (one `video_url` block per request). Measures the full pipeline including
  server-side MP4 decode.
- **`mixed.yaml`** — All three modalities (image + video + audio) in one
  request. Useful for end-to-end smoke tests on a multimodal-capable model.
- **`shared_prefix.yaml`** — Shared-prefix benchmark with one cached image in
  the prefix and one fresh image per request. Exercises the model server's
  prefix-cache hit rate for image content. Uses
  `video.representation: png_frames` to emit a video as a sequence of PNG
  `image_url` frame blocks (deterministic bytes for cache hits). Switch to
  `jpeg_frames` for smaller wire payloads.

## Picking a model

Each config has a `server.model_name` placeholder. Swap it for a model that
actually supports the modalities you're sending — e.g. Qwen2-VL / Qwen3-VL
for image+video, Qwen2-Audio for audio. See
[docs/config.md](../../docs/config.md#multimodal-data-generation) for guidance
on per-model resolution and modality limits.

## Running

From the repo root, with your model server reachable at `server.base_url`:

```bash
inference-perf --config_file examples/multimodal/image.yaml
```

The lifecycle report (`summary_lifecycle_metrics.json`) will include
`throughput.{images,videos,audios}_per_sec` and per-modality distribution
blocks (`image.{count,pixels,bytes,aspect_ratio}` etc.).
