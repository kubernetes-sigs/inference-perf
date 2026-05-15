# `inference_perf.payloads`

Typed Pydantic models describing the data that flows in and out of a benchmark
request — both **pre-flight** (what we plan to send) and **post-flight** (what
was actually measured). These are the structural contracts shared between
`datagen/` (which produces specs), `apis/` (which materializes them into wire
requests and parses responses into metrics), and `reportgen/` (which consumes
the metrics).

The naming convention:

- **`Spec`** — the *plan*. Sampled by datagen pre-flight; describes what we
  intend to send.
- **`RequestBody`** — the *wire body*. The dict that gets serialized to HTTP.
  Named for clarity but kept as a plain `dict[str, Any]` alias (not a class)
  because modeling it would mean materialized image/video/audio bytes living
  on every lifecycle metric.
- **`Metrics`** — the *measurement*. Realized stats of what was actually sent
  / received, recorded post-flight and attached to
  `InferenceInfo.request_metrics` / `InferenceInfo.response_metrics` for the
  per-request lifecycle JSON + reports.

The lowercased word "payload" still appears in `apis/chat.py` as the natural
counterpart to "prefix" in shared-prefix benchmarks (prefix-side vs
payload-side content).

## Layout

Modality is the primary organizing axis. Each per-modality subpackage
co-locates the spec types and the metric types for that modality, mirroring
the field-for-field correspondence between the two top-level aggregators:

```
__init__.py        MultimodalSpec + RequestMetrics + RequestBody + re-exports
image/
  spec/            ImageSpec, ImageRepresentation, +
                   SyntheticImageSpec, PreEncodedImageSpec (stub),
                   RemoteImageSpec (stub), LocalFileImageSpec (stub)
  metrics.py       Image, Images
audio/
  spec/            AudioSpec, + SyntheticAudioSpec,
                   PreEncodedAudioSpec (stub), RemoteAudioSpec (stub),
                   LocalFileAudioSpec (stub)
  metrics.py       Audio, Audios
video/
  spec/            VideoSpec + VideoRepresentation,
                   SyntheticMp4VideoSpec, SyntheticFramesVideoSpec,
                   PreEncodedFramesVideoSpec, RemoteVideoSpec (stub),
                   LocalFileVideoSpec (stub), VideoSpecUnion
  metrics.py       Video, Videos
text/
  metrics.py       Text     (no spec; text-side request fields live
                             on InferenceAPIData subclasses)
```

The two aggregators line up:

```python
class MultimodalSpec(BaseModel):       class RequestMetrics(BaseModel):
    images: List[SyntheticImageSpec]       text: Text
    videos: List[VideoSpecUnion]           image: Optional[Images]
    audios: List[SyntheticAudioSpec]       video: Optional[Videos]
                                           audio: Optional[Audios]
```

## Provenance axis (spec side only)

Each modality's `spec/` subpackage is further factored
**modality-primary / provenance-secondary**:

- **Synthetic** — bytes generated at materialization time from geometry hints.
- **Pre-encoded** — bytes supplied by an upstream dataset loader (e.g.
  ShareGPT4Video frames).
- **Remote** — URL referenced; the server fetches the bytes. `bytes=0` on the
  realized metric.
- **Local file** — bytes read from a disk path at materialization time. Same
  wire shape as pre-encoded; deferred read.

Only synthetic specs are wired into the materializer today (plus
`PreEncodedFramesVideoSpec` for the ShareGPT4Video loader). Remote and local
specs are typed stubs awaiting their materializer branches; their files exist
so future wire-up doesn't require restructuring.

### Where wire-format choices live

For images and frame-based videos, the wire encoding (PNG vs JPEG) is a
`representation: ImageRepresentation` field on the Spec — a *value*, not a
*type*. Subclassing along the encoding axis would create classes that share
all fields and all behavior except a single string, so we keep encoding as a
field and reserve subclassing for axes that genuinely change fields or
materializer paths.

`VideoRepresentation` (`mp4` / `png_frames` / `jpeg_frames`) is the
user-facing wire-format enum used by config (`VideoDatagenConfig`,
`ShareGPT4VideoConfig`). Datagen translates a chosen `VideoRepresentation`
into the appropriate Spec subclass plus an `ImageRepresentation`
`frame_representation` field; Spec classes themselves don't carry
`VideoRepresentation`.

## Lifecycle

1. **Datagen samples** a `MultimodalSpec` and attaches it to the API data
   object (`ChatCompletionAPIData.multimodal_spec`).
2. **The materializer** ([`apis/chat.py`](../apis/chat.py)) reads the Spec,
   produces wire bytes, **and** records realized
   `Image` / `Video` / `Audio` records.
3. **`process_response`** wraps those records in `Images` / `Videos` /
   `Audios` containers and assembles a `RequestMetrics` (also carrying
   `Text` token counts).
4. **Reportgen** aggregates `RequestMetrics` across all requests for the
   summary tables.

The response-side counterpart — `UnaryResponseMetrics` /
`StreamedResponseMetrics` on `InferenceInfo.response_metrics` — is defined in
`inference_perf.apis.base` because it's coupled to streaming semantics
(chunk timings, server usage).

## Why both halves live here

`MultimodalSpec` and `RequestMetrics` are typed data contracts shared across
`datagen/`, `apis/`, and `reportgen/` — so they sit together rather than
hiding inside one of them. Within each modality's directory, the spec and
metric files sit next to each other because the materializer's spec→metric
mapping per modality is the only place they touch; co-locating them makes
that mapping visually obvious.

Note: the two aggregators are not perfectly symmetric. `RequestMetrics`
carries text + per-modality realized stats, while `MultimodalSpec` only
carries the multimodal portion of the plan — text-side request fields
(prompt, max_tokens, model, messages) live on `InferenceAPIData` subclasses
(`ChatCompletionAPIData`, `CompletionAPIData`). The `text/` directory exists
for structural consistency and currently holds only the post-flight `Text`
metric record; there is no `text/spec/` because there are no text-side
specs to put there.
