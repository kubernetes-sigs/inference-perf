# Vendored from llm-d/llm-d-benchmark @ 2c1e326e22c47c0e53028914d72ff360a8ce2e79
# Source: llmdbenchmark/analysis/benchmark_report/schema_v0_2_1.py
# Do not edit; resync from upstream when bumping the BR0.2 schema.
"""
Benchmark report v0.2.1

Additive minor revision of v0.2 that adds optional multi-modal payload
statistics (image / video / audio) to the request aggregates.

Every field introduced here is Optional, so any document valid under v0.2 is
also valid under v0.2.1. v0.2 is imported and extended in place rather than
copied, so the unchanged majority of the schema keeps a single definition and
this file contains only the multi-modal delta plus the containment shims needed
to thread the extended aggregates up to a new report root.

Scope note: this revision covers the results side only (the per-modality stats
the client can derive from the payloads it sent, mirroring the fields emitted by
inference-perf's lifecycle report). A standardized load-side `multimodal`
descriptor on LoadStandardized is deliberately left out of this revision; see
the PR description.
"""

from typing import ClassVar

from pydantic import BaseModel

from .base import (
    UNITS_MEDIA_THROUGHPUT,
    UNITS_MEMORY,
    UNITS_QUANTITY,
    UNITS_RATIO,
    UNITS_TIME,
    Units,
    UnitsValidatedModel,
)
from .schema_v0_2 import (
    MODEL_CONFIG,
    VERSION as VERSION_V02,
    AggregateRequestPerformance as AggregateRequestPerformanceV02,
    AggregateRequests as AggregateRequestsV02,
    AggregateThroughput as AggregateThroughputV02,
    BenchmarkReportV02,
    RequestPerformance as RequestPerformanceV02,
    Results as ResultsV02,
    Run,
    Scenario,
    Statistics,
)

# BenchmarkReport schema version
VERSION = "0.2.1"

# v0.2.1 is a strict additive superset of v0.2; this guards against a future
# v0.2 bump silently drifting out from under the version we extend.
assert VERSION_V02 == "0.2", f"schema_v0_2_1 expects to extend v0.2, found {VERSION_V02}"


###############################################################################
# Per-modality payload statistics
#
# Single-inheritance hierarchy so that fields shared across modalities are
# declared exactly once:
#
#   MediaPayloadStats        count, filesize              (all modalities)
#     └─ VisualPayloadStats  + pixels, aspect_ratio       (image, video)
#         ├─ ImagePayloadStats
#         └─ VideoPayloadStats  + frames
#     └─ AudioPayloadStats   + duration
#
# Adding a modality is a new leaf class plus one field on MultiModalRequests.
###############################################################################


class MediaPayloadStats(UnitsValidatedModel):
    """Payload statistics shared by every media modality.

    All fields are distributions over the individual media instances the client
    sent, derived purely from the request payload.
    """

    model_config = MODEL_CONFIG.copy()

    UNIT_RULES: ClassVar[dict[str, list[Units]]] = {
        "count": UNITS_QUANTITY,
        "filesize": UNITS_MEMORY,
    }

    count: Statistics | None = None
    """Number of media instances of this modality per request."""
    filesize: Statistics | None = None
    """Encoded size per media instance."""


class VisualPayloadStats(MediaPayloadStats):
    """Payload statistics common to pixel-based modalities (image and video)."""

    model_config = MODEL_CONFIG.copy()

    UNIT_RULES: ClassVar[dict[str, list[Units]]] = {
        "pixels": UNITS_QUANTITY,
        "aspect_ratio": UNITS_RATIO,
    }

    pixels: Statistics | None = None
    """Pixel count per media instance (height x width, summed over frames)."""
    aspect_ratio: Statistics | None = None
    """Aspect ratio (width / height) per media instance."""


class ImagePayloadStats(VisualPayloadStats):
    """Image payload statistics."""

    model_config = MODEL_CONFIG.copy()


class VideoPayloadStats(VisualPayloadStats):
    """Video payload statistics."""

    model_config = MODEL_CONFIG.copy()

    UNIT_RULES: ClassVar[dict[str, list[Units]]] = {"frames": UNITS_QUANTITY}

    frames: Statistics | None = None
    """Number of frames per video instance."""


class AudioPayloadStats(MediaPayloadStats):
    """Audio payload statistics."""

    model_config = MODEL_CONFIG.copy()

    UNIT_RULES: ClassVar[dict[str, list[Units]]] = {"duration": UNITS_TIME}

    duration: Statistics | None = None
    """Duration per audio instance."""


class MultiModalRequests(BaseModel):
    """Per-modality request payload statistics for multi-modal workloads."""

    model_config = MODEL_CONFIG.copy()

    image: ImagePayloadStats | None = None
    """Image payload statistics."""
    video: VideoPayloadStats | None = None
    """Video payload statistics."""
    audio: AudioPayloadStats | None = None
    """Audio payload statistics."""


###############################################################################
# Extended request aggregates
###############################################################################


class AggregateRequests(AggregateRequestsV02, UnitsValidatedModel):
    """v0.2 request statistics, plus multi-modal payload details.

    Inherits the v0.2 input/output-length unit checks and adds a declarative
    rule for the new request_size field.
    """

    model_config = MODEL_CONFIG.copy()

    UNIT_RULES: ClassVar[dict[str, list[Units]]] = {"request_size": UNITS_MEMORY}

    request_size: Statistics | None = None
    """Total encoded request size, including all media payloads."""
    multimodal: MultiModalRequests | None = None
    """Per-modality payload statistics."""


class AggregateThroughput(AggregateThroughputV02, UnitsValidatedModel):
    """v0.2 throughput metrics, plus per-modality payload rates."""

    model_config = MODEL_CONFIG.copy()

    UNIT_RULES: ClassVar[dict[str, list[Units]]] = {
        "image_rate": UNITS_MEDIA_THROUGHPUT,
        "video_rate": UNITS_MEDIA_THROUGHPUT,
        "audio_rate": UNITS_MEDIA_THROUGHPUT,
    }

    image_rate: Statistics | None = None
    """Image delivery rate."""
    video_rate: Statistics | None = None
    """Video delivery rate."""
    audio_rate: Statistics | None = None
    """Audio delivery rate."""


###############################################################################
# Containment shims: re-thread the extended aggregates up to a new report root.
# Each class redeclares only the field whose type changed; all other fields are
# inherited from the v0.2 definition.
###############################################################################


class AggregateRequestPerformance(AggregateRequestPerformanceV02):
    """Aggregate performance metrics (v0.2.1 aggregates)."""

    model_config = MODEL_CONFIG.copy()

    requests: AggregateRequests | None = None
    """Aggregate request details."""
    throughput: AggregateThroughput | None = None
    """Aggregate response throughput performance metrics."""


class RequestPerformance(RequestPerformanceV02):
    """Request-level performance metrics (v0.2.1 aggregates)."""

    model_config = MODEL_CONFIG.copy()

    aggregate: AggregateRequestPerformance | None = None
    """Aggregate performance metrics."""


class Results(ResultsV02):
    """Benchmark results (v0.2.1 request performance)."""

    model_config = MODEL_CONFIG.copy()

    request_performance: RequestPerformance | None = None
    """Request-level performance metrics."""


class BenchmarkReportV021(BenchmarkReportV02):
    """Benchmark report v0.2.1."""

    model_config = MODEL_CONFIG.copy()
    model_config["title"] = "Benchmark Report v0.2.1"

    version: str = VERSION
    """Version of the schema."""
    run: Run
    """Benchmark run details."""
    scenario: Scenario | None = None
    """Stack configuration and workload details of experiment."""
    results: Results
    """Experiment results."""
