#!/usr/bin/env python3
# Copyright 2024 The inference-perf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify a synthetic_agentic run by reconstructing its sessions from Jaeger spans.

This is the live-loop verifier: after running a `synthetic_agentic` config with
Jaeger tracing enabled (see examples/otel/run_with_jaeger.sh), query the Jaeger HTTP API,
reconstruct each session from its spans, and assert the replayed sessions match what the
generator intended — every session succeeded (no dangling tool_call_id / 400), and the
per-session event count is what the config's shape implies.

Usage:
    python tools/verify_synthetic_via_jaeger.py \
        --expect-sessions 5 --expect-events 4 --lookback 10m

    # fan-out: don't pin an exact event count, just require success + a minimum
    python tools/verify_synthetic_via_jaeger.py --expect-sessions 5 --min-events 7

Exit code 0 = all assertions passed; 1 = a verification failure; 2 = usage/connection error.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def _peak_concurrency(chat_spans: list[dict]) -> int:
    """Max number of llm.chat.completions spans in-flight at once (evidence of parallel
    sub-agent execution). A sweep of a fan-out tree should show peak > 1; a purely
    sequential (single-agent) session shows peak == 1."""
    events: list[tuple[int, int]] = []
    for s in chat_spans:
        start = s.get("startTime", 0)
        events.append((start, 1))
        events.append((start + s.get("duration", 0), -1))
    # sort by time; process ends (-1) before starts (+1) at identical timestamps
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def fetch_synthetic_sessions(jaeger_url: str, lookback: str, limit: int) -> list[dict]:
    """Return one dict per synthetic session found in Jaeger (session.id starts with 'synthN')."""
    url = f"{jaeger_url}/api/traces?service=inference-perf&limit={limit}&lookback={lookback}"
    try:
        data = json.load(urllib.request.urlopen(url, timeout=15))
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"ERROR: cannot reach Jaeger at {jaeger_url}: {e}", file=sys.stderr)
        sys.exit(2)

    sessions: list[dict] = []
    for trace in data.get("data", []):
        spans = trace.get("spans", [])
        for span in spans:
            if not span["operationName"].startswith("session."):
                continue
            tags = {t["key"]: t["value"] for t in span.get("tags", [])}
            sid = tags.get("session.id", "")
            if not sid.startswith("synthN"):
                continue
            # NOTE: counts/ times all llm.chat.completions spans in this trace. Under the default
            # OTEL_TRACE_PER_STAGE=false, each session is its own trace root, so this is correctly
            # scoped to one session. (Would over-count if per-stage tracing were enabled.)
            chat_spans = [s for s in spans if s["operationName"] == "llm.chat.completions"]
            sessions.append(
                {
                    "sid": sid,
                    "num_events_tag": tags.get("session.num_events"),
                    "chat_spans": len(chat_spans),
                    "status": tags.get("otel.status_code", "OK"),
                    "error": bool(tags.get("error", False)),
                    "start_time": span.get("startTime", 0),
                    # peak concurrent in-flight chat calls (evidence of sub-agent parallelism)
                    "peak_concurrency": _peak_concurrency(chat_spans),
                }
            )
    # Scope to the SINGLE most-recent run. Session ids repeat across runs (synthN0, synthN1, ...)
    # AND a run may have fewer sessions than a prior one, so both "stale sid shadows current" and
    # "extra sid from a bigger prior run leaks in" are possible. A run's sessions all start within
    # a few seconds of each other, while the previous run is far earlier — so cluster on start_time:
    # keep only sessions whose start_time is within `run_window_us` of the newest session.
    if not sessions:
        return []
    run_window_us = 120 * 1_000_000  # 120s: comfortably wider than one run, tighter than run gaps
    newest = max(s["start_time"] for s in sessions)
    in_run = [s for s in sessions if newest - s["start_time"] <= run_window_us]
    # within the run, de-dup by sid keeping the latest (defensive; sids are unique per run)
    by_sid: dict[str, dict] = {}
    for s in in_run:
        prev = by_sid.get(s["sid"])
        if prev is None or s["start_time"] > prev["start_time"]:
            by_sid[s["sid"]] = s
    return list(by_sid.values())


def verify(
    sessions: list[dict],
    expect_sessions: int | None,
    expect_events: int | None,
    min_events: int | None,
    min_peak_concurrency: int | None,
) -> list[str]:
    """Return a list of failure strings (empty = all good)."""
    failures: list[str] = []
    if expect_sessions is not None and len(sessions) != expect_sessions:
        failures.append(f"expected {expect_sessions} synthetic sessions in Jaeger, found {len(sessions)}")

    max_peak_seen = max((s["peak_concurrency"] for s in sessions), default=0)
    for s in sorted(sessions, key=lambda x: x["sid"]):
        if s["status"] != "OK" or s["error"]:
            failures.append(
                f"{s['sid']}: session did not succeed (status={s['status']} error={s['error']}) "
                f"— a dangling tool_call_id or 400 shows up here"
            )
        # the runtime should have executed one chat call per event
        try:
            n_tag = int(s["num_events_tag"]) if s["num_events_tag"] is not None else None
        except (TypeError, ValueError):
            n_tag = None
        if n_tag is not None and s["chat_spans"] != n_tag:
            failures.append(
                f"{s['sid']}: chat_spans={s['chat_spans']} != session.num_events={n_tag} "
                f"(runtime did not execute one call per event)"
            )
        if expect_events is not None and n_tag != expect_events:
            failures.append(f"{s['sid']}: num_events={n_tag} != expected {expect_events}")
        if min_events is not None and (n_tag is None or n_tag < min_events):
            failures.append(f"{s['sid']}: num_events={n_tag} < min expected {min_events} (fan-out did not materialize?)")

    # Sub-agent parallelism check: assert AT LEAST ONE session reached the required peak of
    # concurrent in-flight calls. This is per-run (not per-session) because with
    # fanout_probability<1 some sessions legitimately don't spawn and run sequentially.
    if min_peak_concurrency is not None and max_peak_seen < min_peak_concurrency:
        failures.append(
            f"no session reached peak concurrent in-flight calls >= {min_peak_concurrency} "
            f"(max seen = {max_peak_seen}); sub-agent fan-out is not running in parallel"
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaeger-url", default="http://localhost:16686")
    ap.add_argument("--lookback", default="10m", help="Jaeger lookback window (e.g. 10m, 1h)")
    ap.add_argument("--limit", type=int, default=50, help="max traces to pull")
    ap.add_argument("--expect-sessions", type=int, default=None, help="assert exactly this many synthetic sessions")
    ap.add_argument("--expect-events", type=int, default=None, help="assert each session has exactly this many events")
    ap.add_argument("--min-events", type=int, default=None, help="assert each session has at least this many events")
    ap.add_argument(
        "--min-peak-concurrency",
        type=int,
        default=None,
        help="assert at least one session reached this many concurrent in-flight calls "
        "(evidence that sub-agent fan-out runs in parallel)",
    )
    args = ap.parse_args()

    sessions = fetch_synthetic_sessions(args.jaeger_url, args.lookback, args.limit)
    print(f"synthetic sessions found in Jaeger: {len(sessions)}")
    for s in sorted(sessions, key=lambda x: x["sid"]):
        print(
            f"  {s['sid']}: num_events={s['num_events_tag']} chat_spans={s['chat_spans']} "
            f"peak_concurrency={s['peak_concurrency']} status={s['status']} error={s['error']}"
        )

    failures = verify(sessions, args.expect_sessions, args.expect_events, args.min_events, args.min_peak_concurrency)
    if failures:
        print("\nVERIFICATION FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nVERIFICATION PASSED: all synthetic sessions replayed successfully and match intent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
