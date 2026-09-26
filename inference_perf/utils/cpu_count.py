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
"""CPU-count helper for default worker concurrency.

Uses the process CPU affinity as the primary signal and cgroup CPU quotas
as an additional clamp, falling back to the host CPU count:

    min(affinity, cgroup quota, host CPUs)

Affinity covers restrictions that carry no CFS quota (``--cpuset-cpus``,
``taskset``, kubelet CPU Manager ``static`` policy). Quota covers
``limits.cpu`` / ``--cpus``. The host count clamps both so an oversized
quota (e.g. ``limits.cpu: "64"`` on an 8-core node, which Kubernetes
schedules since only requests are validated against node capacity) cannot
inflate the default worker pool.
"""

import functools
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_QUOTA_US = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_PERIOD_US = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _quota_cpus(quota_str: str, period_str: str) -> Optional[int]:
    """Convert a (quota, period) pair to whole CPUs, or None if unlimited/invalid."""
    try:
        quota_s = quota_str.strip()
        period_s = period_str.strip()
        if quota_s == "max":
            return None
        quota = int(quota_s)
        period = int(period_s)
        if quota <= 0 or period <= 0:
            return None
        return max(1, quota // period)
    except (ValueError, ZeroDivisionError) as e:
        logger.debug("Ignoring invalid CPU quota (%r / %r): %s", quota_str, period_str, e)
        return None


def _parse_cpu_max(text: str) -> Optional[int]:
    """Parse a cgroup v2 ``cpu.max`` payload ("<quota> <period>")."""
    parts = text.split()
    if len(parts) != 2:
        logger.debug("Ignoring malformed cpu.max %r: expected '<quota> <period>'", text)
        return None
    return _quota_cpus(parts[0], parts[1])


def _cgroup_quota_cpus() -> Optional[int]:
    """Return the cgroup CPU quota in whole CPUs, or None when unconstrained."""
    try:
        quota = _parse_cpu_max(CGROUP_V2_CPU_MAX.read_text())
        if quota is not None:
            return quota
    except OSError as e:
        logger.debug("Unable to read %s: %s", CGROUP_V2_CPU_MAX, e)
    try:
        quota = _quota_cpus(CGROUP_V1_QUOTA_US.read_text(), CGROUP_V1_PERIOD_US.read_text())
        if quota is not None:
            return quota
    except OSError as e:
        logger.debug("Unable to read cgroup v1 quota files: %s", e)
    return None


def _affinity_cpus() -> Optional[int]:
    """Return usable CPUs from affinity, or None when unavailable."""
    # TODO(py3.13): replace this shim with os.process_cpu_count().
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:
        return None
    try:
        affinity_count = len(getaffinity(0))
    except (OSError, NotImplementedError) as e:
        logger.debug("Unable to read CPU affinity: %s", e)
        return None
    return affinity_count if affinity_count > 0 else None


@functools.cache
def default_cpu_count() -> int:
    """Return CPUs available to this process for sizing the worker pool.

    ``min(affinity, cgroup quota, host CPUs)`` with a floor of 1. Affinity
    covers pinning without a quota (cpuset, taskset); the quota covers
    ``limits.cpu``/``--cpus``. Unreadable or unconstrained sources fall back
    to the host count.
    """
    host_cpus = max(1, os.cpu_count() or 1)
    candidates: dict[str, int] = {"host": host_cpus}

    affinity = _affinity_cpus()
    if affinity is not None:
        candidates["affinity"] = affinity

    quota = _cgroup_quota_cpus()
    if quota is not None:
        # Kubernetes validates requests but not limits against node capacity,
        # so an oversized quota must not inflate the pool past physical CPUs.
        candidates["cgroup_quota"] = min(quota, host_cpus)

    winner = min(candidates, key=lambda k: candidates[k])
    result = max(1, candidates[winner])
    logger.debug("CPU sources %s -> %d (via %s)", candidates, result, winner)
    return result
