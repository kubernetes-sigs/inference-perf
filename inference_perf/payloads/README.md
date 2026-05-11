# `inference_perf.payloads`

Typed Pydantic models describing the data that flows in and out of a benchmark
request — both **pre-flight** (what we plan to send) and **post-flight** (what
was actually measured). These are the structural contracts shared between
`datagen/` (which produces specs), `apis/` (which materializes them into wire
requests and parses responses into metrics), and `reportgen/` (which consumes
the metrics).

The naming follows the convention:

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

## Files

### `payload.py` — post-flight measurement records

Realized stats of what was *actually sent*, recorded after the request runs
and attached to `InferenceInfo.request_metrics` for the lifecycle metrics +
reports.

- `Text` — input token count for what we sent. Output-side counts live on
  `*ResponseMetrics`, not here — `RequestMetrics` is sent-side only.
- `Image`, `Video`, `Audio` — per-instance realized stats (pixels, byte size,
  aspect ratio, frame count, duration).
- `Images`, `Videos`, `Audios` — request-scoped containers (count + per-instance
  list).
- `RequestMetrics` — the top-level wrapper combining text + optional
  per-modality containers. Lives on `InferenceInfo.metrics`.

The response-side counterpart — `UnaryResponseMetrics` / `StreamedResponseMetrics`
on `InferenceInfo.response_metrics` — is defined in `inference_perf.apis.base`
because it's coupled to streaming semantics (chunk timings, server usage).

### `multimodal_spec.py` — pre-flight request specs

What we plan to send, sampled by datagen and consumed by an API
implementation's `to_request_body`. Carries dimensions, profiles, durations,
and insertion points — but **never raw bytes**: bytes are materialized at
request-build time and don't live on the lifecycle metric.

- `ImageInstanceSpec`, `VideoInstanceSpec`, `AudioInstanceSpec` — per-instance
  spec for one image/video/audio attachment.
- `MultimodalSpec` — request-scoped container of all the per-instance specs.

## Why both live here

`RequestMetrics` and `MultimodalSpec` are typed data contracts shared across
`datagen/`, `apis/`, and `reportgen/` — so they sit together rather than
hiding inside one of them.

Note: the two are not perfectly symmetric. `RequestMetrics` is request-scoped
(text + per-modality realized stats), while `MultimodalSpec` only carries
the multimodal portion of the plan — the text-side request fields (prompt,
max_tokens, model, messages) live on `InferenceAPIData` subclasses
(`ChatCompletionAPIData`, `CompletionAPIData`). A future cleanup could pull
those onto a unified request-scoped spec; until then `MultimodalSpec` is
named for the slice it actually covers.
