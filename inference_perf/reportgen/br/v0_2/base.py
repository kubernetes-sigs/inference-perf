# Vendored from llm-d/llm-d-benchmark @ 2c1e326e22c47c0e53028914d72ff360a8ce2e79
# Source: llmdbenchmark/analysis/benchmark_report/base.py
# Do not edit; resync from upstream when bumping the BR0.2 schema.
"""
Benchmark report base class with common methods.
"""

import json
from enum import StrEnum, auto
from typing import Any, ClassVar

from pydantic import BaseModel, model_validator
import yaml

###############################################################################
# Supported workload generators
###############################################################################


class WorkloadGenerator(StrEnum):
    """
    Enumeration of supported workload generators

    Attributes
        AIPERF: str
            AIPerf
        GUIDELLM: str
            GuideLLM
        INFERENCE_MAX: str
            InferenceMAX
        INFERENCE_PERF: str
            Inference Perf
        VLLM_BENCHMARK: str
            benchmark_serving from vLLM
        NOP: str
            vLLM Load times
    """

    AIPERF = "aiperf"
    GUIDELLM = auto()
    INFERENCE_MAX = "inferencemax"
    INFERENCE_PERF = "inference-perf"
    VLLM_BENCHMARK = "vllm-benchmark"
    NOP = "nop"


###############################################################################
# Units
###############################################################################


class Units(StrEnum):
    """
    Enumeration of units

    Attributes
        COUNT: str
            Count
        MS: str
            Milliseconds
        S: str
            Seconds
        MB: str
            Megabytes
        GB: str
            Gigabytes
        TB: str
            Terabytes
        MIB: str
            Mebibytes
        GIB: str
            Gibibytes
        TIB: str
            Tebibytes
        MBIT_PER_S: str
            Megabbits per second
        GBIT_PER_S: str
            Gigabits per second
        TBIT_PER_S: str
            Terabits per second
        MB_PER_S: str
            Megabytes per second
        GB_PER_S: str
            Gigabytes per second
        TB_PER_S: str
            Terabytes per second
        GIB_PER_S: str
            GiB per second
        MS_PER_TOKEN: str
            Milliseconds per token
        S_PER_TOKEN: str
            Seconds per token
        TOKEN_PER_S: str
            Tokens per second
        WATTS: str
            Watts
    """

    # Quantity
    COUNT = auto()
    PIXELS = auto()
    # Portion
    PERCENT = auto()
    FRACTION = auto()
    # Ratio (unbounded; unlike a portion, may exceed 1, e.g. aspect ratio)
    RATIO = auto()
    # Time
    MS = auto()
    S = auto()
    # Memory
    BYTES = "bytes"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    MIB = "MiB"
    GIB = "GiB"
    TIB = "TiB"
    # Bandwidth
    MBIT_PER_S = "Mbit/s"
    GBIT_PER_S = "Gbit/s"
    TBIT_PER_S = "Tbit/s"
    GIB_PER_S = "GiB/s"

    MB_PER_S = "MB/s"
    GB_PER_S = "GB/s"
    TB_PER_S = "TB/s"
    # Generation latency
    MS_PER_TOKEN = "ms/token"
    S_PER_TOKEN = "s/token"
    # Generation throughput
    TOKEN_PER_S = "tokens/s"
    # Request throughput
    QUERY_PER_S = "queries/s"
    # Media throughput (per-modality payload rates)
    IMAGE_PER_S = "images/s"
    VIDEO_PER_S = "videos/s"
    AUDIO_PER_S = "audios/s"
    # Power
    WATTS = "Watts"


# Lists of compatible units for a particular application
UNITS_QUANTITY = [Units.COUNT, Units.PIXELS]
UNITS_PORTION = [Units.PERCENT, Units.FRACTION]
UNITS_RATIO = [Units.RATIO]
UNITS_TIME = [Units.MS, Units.S]
UNITS_MEMORY = [
    Units.BYTES,
    Units.MB,
    Units.GB,
    Units.TB,
    Units.MIB,
    Units.GIB,
    Units.TIB,
]
UNITS_BANDWIDTH = [
    Units.MBIT_PER_S,
    Units.GBIT_PER_S,
    Units.TBIT_PER_S,
    Units.MB_PER_S,
    Units.GB_PER_S,
    Units.TB_PER_S,
]
UNITS_GEN_LATENCY = [Units.MS_PER_TOKEN, Units.S_PER_TOKEN]
UNITS_GEN_THROUGHPUT = [Units.TOKEN_PER_S]
UNITS_REQUEST_THROUGHPUT = [Units.QUERY_PER_S]
UNITS_MEDIA_THROUGHPUT = [Units.IMAGE_PER_S, Units.VIDEO_PER_S, Units.AUDIO_PER_S]
UNITS_POWER = [Units.WATTS]


###############################################################################
# Declarative units validation
###############################################################################


class UnitsValidatedModel(BaseModel):
    """Base model that validates ``Statistics`` field units declaratively.

    Instead of hand-writing a ``check_units`` method per class, a subclass sets
    ``UNIT_RULES``, mapping a field name to the list of units allowed for that
    field's ``Statistics.units``. A single inherited validator checks every
    rule declared anywhere in the class's MRO, so each subclass only declares
    the fields it introduces; ``None`` (unset Optional) fields are skipped.

    The validator is named distinctly from any ``check_units`` so it co-runs
    with, rather than shadows, a hand-written validator inherited from another
    base under multiple inheritance.
    """

    # field name -> allowed Units. Merged across the MRO at validation time.
    UNIT_RULES: ClassVar[dict[str, list[Units]]] = {}

    @model_validator(mode="after")
    def validate_declared_units(self):
        merged: dict[str, list[Units]] = {}
        # Most-derived first, so a subclass rule overrides a base rule.
        for klass in type(self).__mro__:
            for field, allowed in klass.__dict__.get("UNIT_RULES", {}).items():
                merged.setdefault(field, allowed)
        for field, allowed in merged.items():
            stat = getattr(self, field, None)
            if stat is not None and stat.units not in allowed:
                raise ValueError(f'Invalid units "{stat.units}" for "{field}", must be one of: {" ".join(allowed)}')
        return self


###############################################################################
# Base benchmark report class
###############################################################################


class BenchmarkReport(BaseModel):
    """Common base class for a benchmark report."""

    def dump(self) -> dict[str, Any]:
        """Convert BenchmarkReport to dict.

        Returns:
            dict: Defined fields of BenchmarkReport.
        """
        return self.model_dump(
            mode="json",
            exclude_none=True,
            by_alias=True,
        )

    def export_json(self, filename) -> None:
        """Save BenchmarkReport to JSON file.

        Args:
            filename: File to save BenchmarkReport to.
        """
        with open(filename, "w") as file:
            json.dump(self.dump(), file, indent=2)

    def export_yaml(self, filename) -> None:
        """Save BenchmarkReport to YAML file.

        Args:
            filename: File to save BenchmarkReport to.
        """
        with open(filename, "w") as file:
            yaml.dump(self.dump(), file, indent=2)

    def get_json_str(self) -> str:
        """Make a JSON string for BenchmarkReport.

        Returns:
            str: JSON string.
        """
        return json.dumps(self.dump(), indent=2)

    def get_yaml_str(self) -> str:
        """Make a YAML string for BenchmarkReport.

        Returns:
            str: YAML string.
        """
        return yaml.dump(self.dump(), indent=2)
