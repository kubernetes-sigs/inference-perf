# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``inference_perf.utils.cpu_count``."""

from collections.abc import Generator
from pathlib import Path

import pytest

from inference_perf.utils.cpu_count import _parse_cpu_max, _quota_cpus, default_cpu_count


@pytest.fixture(autouse=True)
def _clear_cpu_count_cache() -> Generator[None, None, None]:
    default_cpu_count.cache_clear()
    yield
    default_cpu_count.cache_clear()


def _isolate_cgroup_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point quota file constants at tmp paths so tests are hermetic on any host."""
    v2 = tmp_path / "cpu.max"
    monkeypatch.setattr("inference_perf.utils.cpu_count.CGROUP_V2_CPU_MAX", v2)
    monkeypatch.setattr("inference_perf.utils.cpu_count.CGROUP_V1_QUOTA_US", tmp_path / "no-quota")
    monkeypatch.setattr("inference_perf.utils.cpu_count.CGROUP_V1_PERIOD_US", tmp_path / "no-period")
    return v2


def _fix_cpus(monkeypatch: pytest.MonkeyPatch, affinity: int, host: int = 32) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: host)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(affinity)), raising=False)


@pytest.mark.parametrize(
    ("quota", "period", "expected"),
    [
        ("200000", "100000", 2),
        ("max", "100000", None),
        ("-1", "100000", None),
        ("0", "100000", None),
        ("200000", "0", None),
        ("bogus", "100000", None),
    ],
)
def test_quota_cpus_parsing(quota: str, period: str, expected: int | None) -> None:
    assert _quota_cpus(quota, period) == expected


def test_parse_cpu_max() -> None:
    assert _parse_cpu_max("200000 100000\n") == 2
    assert _parse_cpu_max("max 100000\n") is None
    assert _parse_cpu_max("garbage\n") is None
    # A zero period must fall back, never raise (ZeroDivisionError escapes ValueError handlers).
    assert _parse_cpu_max("100000 0\n") is None


def test_default_cpu_count_uses_quota(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v2 = _isolate_cgroup_files(monkeypatch, tmp_path)
    v2.write_text("200000 100000\n")
    _fix_cpus(monkeypatch, affinity=32)
    assert default_cpu_count() == 2


def test_default_cpu_count_falls_back_without_quota(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v2 = _isolate_cgroup_files(monkeypatch, tmp_path)
    v2.write_text("max 100000\n")
    _fix_cpus(monkeypatch, affinity=8)
    assert default_cpu_count() == 8


def test_default_cpu_count_missing_files_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_cgroup_files(monkeypatch, tmp_path)
    _fix_cpus(monkeypatch, affinity=8)
    assert default_cpu_count() == 8


def test_default_cpu_count_clamped_to_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # limits.cpu above node capacity must not inflate the pool past physical CPUs.
    v2 = _isolate_cgroup_files(monkeypatch, tmp_path)
    v2.write_text("6400000 100000\n")
    _fix_cpus(monkeypatch, affinity=64, host=8)
    assert default_cpu_count() == 8


def test_default_cpu_count_affinity_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v2 = _isolate_cgroup_files(monkeypatch, tmp_path)
    v2.write_text("800000 100000\n")
    _fix_cpus(monkeypatch, affinity=2)
    assert default_cpu_count() == 2


def test_default_cpu_count_uses_v1_quota(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_cgroup_files(monkeypatch, tmp_path)  # v2 file absent -> falls through to v1
    quota = tmp_path / "cpu.cfs_quota_us"
    period = tmp_path / "cpu.cfs_period_us"
    quota.write_text("300000\n")
    period.write_text("100000\n")
    monkeypatch.setattr("inference_perf.utils.cpu_count.CGROUP_V1_QUOTA_US", quota)
    monkeypatch.setattr("inference_perf.utils.cpu_count.CGROUP_V1_PERIOD_US", period)
    _fix_cpus(monkeypatch, affinity=32)
    assert default_cpu_count() == 3
