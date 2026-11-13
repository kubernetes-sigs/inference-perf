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
"""The ``aiperf profile`` rows whose meaning differs between v0.12.0 and v0.13.0.

Each test runs the same argv at both verified tags, so a v0.13.0 row cannot
leak into a v0.12.0 conversion and the reverse.
"""

from typing import List

import pytest

from inference_perf.tools.convert import convert_aiperf_profile
from inference_perf.tools.convert.model import Conversion, PeerUsageError
from inference_perf.tools.convert.versions import AIPERF_PROFILE_PINNED, AIPERF_PROFILE_VERSIONS

V012 = "v0.12.0"
V013 = "v0.13.0"
BASE = ["--model", "m", "--osl", "64", "--num-dataset-entries", "10"]


# Converts BASE plus extra args at one tag, e.g. at("v0.13.0", "--reset-kv-cache").
def at(version: str, *extra: str) -> Conversion:
    return convert_aiperf_profile(BASE + list(extra), version)


# The registry keeps v0.12.0 next to the new pin. Expects both tags to
# resolve and the pin to read "v0.13.0".
def test_pin_moved_and_v0_12_0_stays_verified() -> None:
    assert AIPERF_PROFILE_PINNED == V013
    assert sorted(AIPERF_PROFILE_VERSIONS) == [V012, V013]


# The completions parity argv (rate 40, 600 requests, ISL 128, OSL 64) at
# each tag. Expects the same config at both, since no row it touches moved.
def test_unchanged_rows_emit_the_same_config_at_both_tags() -> None:
    argv = ["--model", "m", "--endpoint-type", "completions", "--streaming", "--request-rate", "40"]
    argv += ["--request-count", "600", "--isl", "128", "--osl", "64", "--num-dataset-entries", "600"]
    old = convert_aiperf_profile(argv, V012)
    new = convert_aiperf_profile(argv, V013)
    assert old.config is not None and new.config is not None
    assert old.config.model_dump(exclude_unset=True) == new.config.model_dump(exclude_unset=True)


@pytest.mark.parametrize(
    "extra",
    [
        ["--per-chunk-usage"],
        ["--random-corpus-style", "vllm"],
        ["--random-range-ratio", "0.5"],
        ["--reset-kv-cache"],
        ["--reset-kv-cache-path", "/flush"],
        ["--reset-kv-cache-timeout-seconds", "5"],
        ["--server-profiler"],
        ["--server-profiler-start-path", "/start"],
        ["--server-profiler-stop-path", "/stop"],
        ["--server-profiler-timeout-seconds", "5"],
        ["--system-prompt", "be brief"],
        ["--system-prompt-file", "prompt.txt"],
    ],
)
# One of the twelve flags v0.13.0 adds, e.g. --reset-kv-cache. At v0.12.0 it
# refuses as unknown. At v0.13.0 no refusal calls it unknown.
def test_flags_added_in_v0_13_0_are_unknown_at_v0_12_0(extra: List[str]) -> None:
    old = at(V012, *extra)
    assert old.refusals == [f"{extra[0]}: not in the verified aiperf profile v0.12.0 flag surface"]
    new = at(V013, *extra)
    assert not any("flag surface" in refusal for refusal in new.refusals), new.refusals


# --bogus at each tag. Expects the unknown-flag refusal to name the tag the
# argv was checked against.
def test_unknown_flag_refusal_names_the_tag() -> None:
    assert at(V012, "--bogus").refusals == ["--bogus: not in the verified aiperf profile v0.12.0 flag surface"]
    assert at(V013, "--bogus").refusals == ["--bogus: not in the verified aiperf profile v0.13.0 flag surface"]


# --endpoint-type audio_transcription. v0.12.0 has no such plugin, so it is
# outside the registry. v0.13.0 registers it, and it has no API type.
def test_audio_transcription_joins_the_registry_at_v0_13_0() -> None:
    old = at(V012, "--endpoint-type", "audio_transcription")
    assert old.refusals == ["--endpoint-type: 'audio_transcription' is not in the verified v0.12.0 endpoint registry"]
    new = at(V013, "--endpoint-type", "audio_transcription")
    assert new.config is None
    assert any("'audio_transcription' has no API type" in refusal for refusal in new.refusals), new.refusals


# --transport http at each tag. Expects the drop note to name the tag whose
# transport registry holds only http.
def test_transport_drop_names_the_tag() -> None:
    assert "--transport (only http is registered at v0.12.0)" in at(V012, "--transport", "http").dropped
    assert "--transport (only http is registered at v0.13.0)" in at(V013, "--transport", "http").dropped


# --isl 128 at each tag. Both emit a fixed 128-token input. Only v0.13.0, which
# subtracts the BOS from the mean, records that --isl is the server-side count.
def test_isl_special_token_reading_is_recorded_at_v0_13_0_only() -> None:
    old = at(V012, "--isl", "128")
    new = at(V013, "--isl", "128")
    assert old.config is not None and new.config is not None
    assert old.config.data.input_distribution == new.config.data.input_distribution
    assert not any(item.startswith("--isl:") for item in old.assumptions)
    assert any(item.startswith("--isl: aiperf v0.13.0 subtracts") for item in new.assumptions), new.assumptions


@pytest.mark.parametrize("extra", [[], ["--session-header", "X-Session"]])
# A bare run and one with --session-header X-Session. Only v0.13.0 adds the
# no-equivalent note for the X-Session-Affinity header it sends by default.
def test_session_affinity_header_note_at_v0_13_0_only(extra: List[str]) -> None:
    old = at(V012, *extra)
    new = at(V013, *extra)
    assert old.config is not None and new.config is not None
    assert not any("X-Session-Affinity" in item for item in old.no_equivalents)
    assert any(item.startswith("X-Session-Affinity: aiperf v0.13.0") for item in new.no_equivalents)


# --session-header X-Session-Affinity at v0.13.0. The flag already maps that
# header to api.session_id_header_key, so no "sends no such header" note.
def test_session_header_naming_the_affinity_header_skips_the_note() -> None:
    conversion = at(V013, "--session-header", "X-Session-Affinity")
    assert conversion.config is not None
    assert conversion.config.api.session_id_header_key == "X-Session-Affinity"
    assert not any(item.startswith("X-Session-Affinity:") for item in conversion.no_equivalents)


@pytest.mark.parametrize("name", ["X-Session-Affinity", "x-session-affinity"])
# --header <name>:pin with X-B:b. v0.12.0 sends both headers. v0.13.0 overwrites
# the affinity one with the correlation id, so only X-B is written.
def test_affinity_header_from_the_argv_drops_at_v0_13_0(name: str) -> None:
    old = at(V012, "--header", f"{name}:pin", "X-B:b")
    assert old.config is not None
    assert old.config.api.headers == {name: "pin", "X-B": "b"}
    new = at(V013, "--header", f"{name}:pin", "X-B:b")
    assert new.config is not None
    assert new.config.api.headers == {"X-B": "b"}
    assert f"--header {name} (aiperf v0.13.0 replaces it with the correlation id)" in new.dropped


# --per-chunk-usage at v0.13.0 refuses (it changes the chat payload and the ITL
# divisor). --per-chunk-usage=false is inert and drops.
def test_per_chunk_usage_refuses_and_its_negation_drops() -> None:
    refused = at(V013, "--endpoint-type", "chat", "--streaming", "--use-server-token-count", "--per-chunk-usage")
    assert refused.config is None
    assert any(
        refusal.startswith("--per-chunk-usage:") and "continuous_usage_stats" in refusal for refusal in refused.refusals
    )
    inert = at(V013, "--per-chunk-usage=false")
    assert inert.config is not None
    assert "--per-chunk-usage (explicitly disabled; inert)" in inert.dropped


@pytest.mark.parametrize(
    ("extra", "path"),
    [
        (["--reset-kv-cache"], "/reset_prefix_cache"),
        (["--reset-kv-cache-path", "/flush"], "/flush"),
        (["--reset-kv-cache-timeout-seconds", "5"], "/reset_prefix_cache"),
    ],
)
# The switch alone, a path alone (/flush), or a timeout alone each turn the
# hook on in aiperf. Expects a config, the path in the note, and the flag dropped.
def test_reset_kv_cache_converts_with_a_note(extra: List[str], path: str) -> None:
    conversion = at(V013, *extra)
    assert conversion.config is not None
    assert any(item.startswith(f"KV-cache reset: aiperf POSTs {path} ") for item in conversion.no_equivalents)
    assert extra[0] in conversion.dropped


# --server-profiler-stop-path /halt alone turns the profiler on. Expects the
# default /start_profile and the given /halt in the note.
def test_server_profiler_converts_with_a_note() -> None:
    conversion = at(V013, "--server-profiler-stop-path", "/halt")
    assert conversion.config is not None
    assert any(
        item.startswith("server profiler: aiperf POSTs /start_profile before each profiling phase and /halt after")
        for item in conversion.no_equivalents
    ), conversion.no_equivalents
    assert "--server-profiler-stop-path" in conversion.dropped


@pytest.mark.parametrize("path", ["flush", "http://host/flush"])
# A hook path aiperf's require_relative_path rejects: no leading slash, or a
# full URL. Expects a usage error naming the flag.
def test_control_hook_path_aiperf_rejects_is_a_usage_error(path: str) -> None:
    with pytest.raises(PeerUsageError, match="--reset-kv-cache-path must be a relative path"):
        at(V013, "--reset-kv-cache-path", path)


@pytest.mark.parametrize(
    ("extra", "reason_fragment"),
    [
        (["--random-range-ratio", "0.5"], "ratio window"),
        (["--system-prompt", "be brief"], "verbatim system prompt"),
        (["--system-prompt-file", "prompt.txt"], "verbatim system prompt"),
    ],
)
# A v0.13.0 flag that shapes the prompt, e.g. --random-range-ratio 0.5.
# Expects no config and a refusal carrying the named reason.
def test_v0_13_0_prompt_shaping_flags_refuse(extra: List[str], reason_fragment: str) -> None:
    conversion = at(V013, *extra)
    assert conversion.config is None
    assert any(refusal.startswith(f"{extra[0]}: ") and reason_fragment in refusal for refusal in conversion.refusals)


# --random-corpus-style sglang without a ratio only picks the random corpus
# token pool. Expects a config and the flag dropped.
def test_random_corpus_style_drops() -> None:
    conversion = at(V013, "--random-corpus-style", "sglang")
    assert conversion.config is not None
    assert any(item.startswith("--random-corpus-style (") for item in conversion.dropped)
