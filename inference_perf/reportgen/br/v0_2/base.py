# Vendored from llm-d/llm-d-benchmark @ e0b95468e27eff59e09d70275044d6c8a17890b7
# Source: llmdbenchmark/analysis/benchmark_report/base.py
# Do not edit; resync from upstream when bumping the BR0.2 schema.
"""
Benchmark report base class with common methods.
"""

import json
from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel
import yaml

###############################################################################
# Supported workload generators
###############################################################################


class WorkloadGenerator(StrEnum):
    """
    Enumeration of supported workload generators

    Attributes
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
    # Portion
    PERCENT = auto()
    FRACTION = auto()
    # Time
    MS = auto()
    S = auto()
    # Memory
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
    # Power
    WATTS = "Watts"


# Lists of compatible units for a particular application
UNITS_QUANTITY = [Units.COUNT]
UNITS_PORTION = [Units.PERCENT, Units.FRACTION]
UNITS_TIME = [Units.MS, Units.S]
UNITS_MEMORY = [Units.MB, Units.GB, Units.TB, Units.MIB, Units.GIB, Units.TIB]
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
UNITS_POWER = [Units.WATTS]

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
