from typing import Any
from typing import cast

import pytest
from inference_perf.reportgen.base import summarize_requests, summarize_prompt_token_usage, ReportGenerator
from inference_perf.apis.base import (
    RequestLifecycleMetric,
    InferenceInfo,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
    SessionLifecycleMetric,
)
from inference_perf.config.reportgen.config import RequestLifecycleMetricsReportConfig
from inference_perf.payloads import RequestMetrics, Text

# The percentile list real reports use; the token-usage metrics must surface all of these.
DEFAULT_PERCENTILES = RequestLifecycleMetricsReportConfig().percentiles


def test_summarize_requests_tpot_calculation() -> None:
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(output_tokens=10, output_token_times=[1.0, 2.0, 3.0]),
    )

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="test_request", info=info, error=None
    )

    # Duration = 3.0 - 1.0 = 2.0
    # Actual tokens = 10
    # Expected TPOT = 2.0 / (10 - 1) = 2.0 / 9 = 0.222...

    result = summarize_requests([metric], [50])

    assert result is not None
    successes = result.successes
    assert "latency" in successes
    assert "time_per_output_token" in successes["latency"]
    tpot_summary = successes["latency"]["time_per_output_token"]
    assert tpot_summary["mean"] == pytest.approx(2.0 / 9.0)


def test_summarize_requests_tpot_fallback() -> None:
    # Test fallback when output_tokens is not available or <= 1
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(output_tokens=0, output_token_times=[1.0, 2.0, 3.0]),
    )

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="test_request", info=info, error=None
    )

    # Duration = 3.0 - 1.0 = 2.0
    # Count = 3

    result = summarize_requests([metric], [50])

    assert result is not None
    successes = result.successes
    tpot_summary = successes["latency"]["time_per_output_token"]
    assert tpot_summary is None


def test_summarize_requests_with_chunks() -> None:
    from unittest.mock import Mock

    mock_tokenizer = Mock()
    # Assume 1 chunk has 2 tokens, another has 3 tokens
    mock_tokenizer.count_tokens.side_effect = lambda text, **kwargs: 2 if "hello" in text else (3 if "world" in text else 0)

    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(
            # output_tokens is the API layer's whole-message count; chunks drive timing (TPOT/TTFT).
            output_tokens=5,
            response_chunks=['{"choices": [{"text": "hello"}]}', '{"choices": [{"text": "world"}]}'],
            chunk_times=[1.0, 2.0],
        ),
    )

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="test_request", info=info, error=None
    )

    result = summarize_requests([metric], [50], tokenizer=mock_tokenizer)

    assert result is not None
    successes = result.successes

    tpot_summary = successes["latency"]["time_per_output_token"]
    assert tpot_summary["mean"] == pytest.approx(0.25)

    ttft_summary = successes["latency"]["time_to_first_token"]
    assert ttft_summary["mean"] == pytest.approx(1.0)


def test_summarize_requests_multiple_tokens_same_timestamp() -> None:
    from unittest.mock import Mock

    mock_tokenizer = Mock()
    mock_tokenizer.count_tokens.side_effect = lambda text, **kwargs: 3 if "hello" in text else 0

    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(response_chunks=['{"choices": [{"text": "hello"}]}'], chunk_times=[1.0]),
    )

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="test_request", info=info, error=None
    )

    summarize_requests([metric], [50], tokenizer=mock_tokenizer)

    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.output_token_times == [1.0, 1.0, 1.0]


def test_itl_not_inflated_by_per_chunk_bos() -> None:
    """Regression for the ITL half of #564, using the real
    neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 tokenizer, which prepends a BOS
    on every call. Re-tokenizing each streamed chunk with special tokens adds
    one phantom token per chunk (doubling len(output_token_times) and halving
    ITL); counting with add_special_tokens=False keeps the per-token timestamp
    series at the true token count, so ITL lands on the same basis as TPOT.
    """
    import json
    from inference_perf.config import CustomTokenizerConfig
    from inference_perf.utils.custom_tokenizer import CustomTokenizer

    try:
        tokenizer = CustomTokenizer(
            CustomTokenizerConfig(pretrained_model_name_or_path="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")
        )
    except Exception as e:  # offline CI / HF hub unreachable
        pytest.skip(f"real tokenizer unavailable: {e}")
    hf = tokenizer.get_tokenizer()

    # Build a one-token-per-chunk stream by decoding each real token id back to text.
    text = "The quick brown fox jumps over the lazy dog"
    ids = hf.encode(text, add_special_tokens=False)
    # transformers>=5.13 types decode() as str | list[str] because it also accepts batched
    # ids; a flat list[int] is the single-sequence form and always decodes to one str.
    chunk_texts = [cast(str, hf.decode([i])) for i in ids]
    n = len(ids)

    # Sanity: with the BOS the per-chunk sum is inflated by exactly one token per
    # chunk (the bug); without it, it matches the whole-message count (the fix).
    assert sum(tokenizer.count_tokens(c, add_special_tokens=True) for c in chunk_texts) == n + len(chunk_texts)
    assert sum(tokenizer.count_tokens(c, add_special_tokens=False) for c in chunk_texts) == n

    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(
            output_tokens=n,  # whole-message count
            response_chunks=[json.dumps({"choices": [{"text": t}]}) for t in chunk_texts],
            chunk_times=[float(i + 1) for i in range(n)],  # 1s between tokens
            server_usage={"completion_tokens": n},
        ),
    )
    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None)

    result = summarize_requests([metric], [50], tokenizer=tokenizer)

    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    # One timestamp per real token, not one-per-token-plus-a-BOS-per-chunk.
    assert len(metric.info.response_metrics.output_token_times) == n

    # ITL is now consistent with TPOT: both average 1.0s/token over the n-1 gaps.
    itl_mean = result.successes["latency"]["inter_token_latency"]["mean"]
    tpot_mean = result.successes["latency"]["time_per_output_token"]["mean"]
    assert itl_mean == pytest.approx(1.0)
    assert tpot_mean == pytest.approx(1.0)
    # Client per-chunk count now agrees with the server's completion_tokens.
    assert result.successes["token_count_mismatches"] == 0


def test_summarize_requests_surfaces_output_token_usage() -> None:
    """successes.output_tokens reports the server's completion_tokens, summed.

    Mirrors the input-side prompt_tokens metric (PR #473): the exact server
    count surfaced separately from the client-side output_len distribution.
    """
    metrics = []
    # Values (0, 100) make every percentile p interpolate exactly to p, so the
    # full default percentile list maps to a known dict.
    for completion_tokens in (0, 100):
        info = InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=completion_tokens,
                server_usage={"prompt_tokens": 10, "completion_tokens": completion_tokens},
            ),
        )
        metrics.append(
            RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None)
        )

    result = summarize_requests(metrics, DEFAULT_PERCENTILES)

    expected = {"total": 100.0, "mean": 50.0, "min": 0.0, "max": 100.0}
    for p in DEFAULT_PERCENTILES:
        expected["median" if p == 50 else f"p{p:g}"] = float(p)
    assert result.successes["output_tokens"] == pytest.approx(expected)


def test_summarize_requests_token_usage_propagates_percentile_keys() -> None:
    """The requested percentiles surface on both server-sourced token metrics.

    output_tokens / prompt_tokens carry mean/min/max plus one key per
    requested percentile (p50 -> 'median'), same as output_len. Locks in
    that whatever percentile list the run passes feeds these metrics too.
    """
    metrics = []
    for completion_tokens in range(1, 101):  # 1..100
        info = InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=completion_tokens,
                server_usage={"prompt_tokens": 10, "completion_tokens": completion_tokens},
            ),
        )
        metrics.append(
            RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None)
        )

    result = summarize_requests(metrics, [0.1, 1, 5, 50, 90, 99, 99.9])

    expected_keys = {"mean", "min", "max", "p0.1", "p1", "p5", "median", "p90", "p99", "p99.9"}
    assert expected_keys <= result.successes["output_tokens"].keys()
    assert expected_keys <= result.successes["prompt_tokens"].keys()
    # p50 is keyed as 'median', never 'p50'.
    assert "p50" not in result.successes["output_tokens"]
    # Distribution keys mirror output_len exactly (same summarize() call).
    assert result.successes["output_tokens"].keys() - {"total"} == result.successes["output_len"].keys()


def test_summarize_requests_output_tokens_falls_back_to_client_count() -> None:
    """Without server usage, output_tokens.total falls back to the client count."""
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=10)),
        response_metrics=StreamedResponseMetrics(output_tokens=4),  # no server_usage
    )
    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None)

    # Single value: every percentile collapses to it.
    result = summarize_requests([metric], DEFAULT_PERCENTILES)

    expected = {"total": 4.0, "mean": 4.0, "min": 4.0, "max": 4.0}
    for p in DEFAULT_PERCENTILES:
        expected["median" if p == 50 else f"p{p:g}"] = 4.0
    assert result.successes["output_tokens"] == pytest.approx(expected)


def test_output_len_not_recomputed_from_chunks() -> None:
    """Regression for issue #564: output_len keeps the API layer's whole-message
    output_tokens and is NOT re-derived by summing per-chunk re-tokenizations
    (which inflate the count, e.g. a BOS added per chunk)."""
    from unittest.mock import Mock

    mock_tokenizer = Mock()
    # Each chunk re-tokenizes to 2, so a per-chunk sum would be 4 (double).
    mock_tokenizer.count_tokens.side_effect = lambda text, **kwargs: 2 if text else 0

    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(
            output_tokens=2,  # API layer's whole-message count
            response_chunks=['{"choices": [{"text": "a"}]}', '{"choices": [{"text": "b"}]}'],
            chunk_times=[1.0, 2.0],
        ),
    )
    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None)

    result = summarize_requests([metric], [50], tokenizer=mock_tokenizer)

    # output_len stays the whole-message count (2), not the per-chunk sum (4).
    assert result.successes["output_len"]["mean"] == pytest.approx(2.0)


def test_summarize_requests_token_mismatch() -> None:
    from unittest.mock import Mock

    mock_tokenizer = Mock()
    mock_tokenizer.count_tokens.side_effect = lambda text, **kwargs: 2 if "hello" in text else (3 if "world" in text else 0)

    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(
            response_chunks=[
                '{"choices": [{"text": "hello"}]}',
                '{"choices": [{"text": "world"}]}',
            ],
            chunk_times=[1.0, 2.0],
            server_usage={"completion_tokens": 6},
        ),
    )

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="test_request", info=info, error=None
    )

    result = summarize_requests([metric], [50], tokenizer=mock_tokenizer)

    assert result is not None
    successes = result.successes
    assert successes["token_count_mismatches"] == 1


def test_use_server_output_tokens_flag_switches_tpot_ntpot_divisor() -> None:
    """The flag normalizes TPOT/NTPOT by the server's completion_tokens instead of
    the client re-tokenized count, while both raw counts stay reported."""

    def make_metric() -> RequestLifecycleMetric:
        info = InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=10,  # client re-tokenized count
                output_token_times=[1.0, 3.0],  # duration 2.0s
                server_usage={"completion_tokens": 5},  # exact server count
            ),
        )
        return RequestLifecycleMetric(
            scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None
        )

    # Default: divide by the client count (10).
    default = summarize_requests([make_metric()], [50])
    assert default.successes["latency"]["time_per_output_token"]["mean"] == pytest.approx(2.0 / 9.0)
    assert default.successes["latency"]["normalized_time_per_output_token"]["mean"] == pytest.approx(10.0 / 10.0)

    # Flag on: divide by the server count (5).
    server = summarize_requests([make_metric()], [50], use_server_output_tokens=True)
    assert server.successes["latency"]["time_per_output_token"]["mean"] == pytest.approx(2.0 / 4.0)
    assert server.successes["latency"]["normalized_time_per_output_token"]["mean"] == pytest.approx(10.0 / 5.0)

    # Both raw counts remain reported regardless of the flag: output_len is the
    # client count, output_tokens is the server count (a single request, so its
    # per-request distribution collapses to that count).
    assert server.successes["output_len"]["mean"] == pytest.approx(10.0)
    assert server.successes["output_tokens"] == pytest.approx(
        {"total": 5.0, "mean": 5.0, "min": 5.0, "max": 5.0, "median": 5.0}
    )


def test_use_server_output_tokens_switches_output_throughput() -> None:
    """The aggregate output/total tokens_per_sec honor the flag: server
    completion_tokens when on, client re-tokenized output_tokens when off."""

    def make_metric() -> RequestLifecycleMetric:
        info = InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=10,  # client re-tokenized count
                output_token_times=[1.0, 3.0],
                server_usage={"completion_tokens": 5},  # exact server count
            ),
        )
        return RequestLifecycleMetric(
            scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None
        )

    # total_time = 10.0s (max end - min start).
    default = summarize_requests([make_metric()], [50])
    assert default.successes["throughput"]["output_tokens_per_sec"] == pytest.approx(10.0 / 10.0)
    assert default.successes["throughput"]["total_tokens_per_sec"] == pytest.approx((5.0 + 10.0) / 10.0)

    server = summarize_requests([make_metric()], [50], use_server_output_tokens=True)
    assert server.successes["throughput"]["output_tokens_per_sec"] == pytest.approx(5.0 / 10.0)
    assert server.successes["throughput"]["total_tokens_per_sec"] == pytest.approx((5.0 + 5.0) / 10.0)


def test_use_server_output_tokens_works_for_non_streaming() -> None:
    """The flag applies to unary (non-streaming) responses too: NTPOT and the
    output throughput use the server's completion_tokens when it's available."""

    def make_metric() -> RequestLifecycleMetric:
        info = InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=UnaryResponseMetrics(
                output_tokens=10,  # client re-tokenized count
                server_usage={"completion_tokens": 5},  # exact server count
            ),
        )
        return RequestLifecycleMetric(
            scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None
        )

    # NTPOT = (end - start) / output_tokens over all successful requests.
    default = summarize_requests([make_metric()], [50])
    assert default.successes["latency"]["normalized_time_per_output_token"]["mean"] == pytest.approx(10.0 / 10.0)
    assert default.successes["throughput"]["output_tokens_per_sec"] == pytest.approx(10.0 / 10.0)

    server = summarize_requests([make_metric()], [50], use_server_output_tokens=True)
    assert server.successes["latency"]["normalized_time_per_output_token"]["mean"] == pytest.approx(10.0 / 5.0)
    assert server.successes["throughput"]["output_tokens_per_sec"] == pytest.approx(5.0 / 10.0)


def test_use_server_output_tokens_switches_goodput_token_total() -> None:
    """Goodput's token_goodput counts server completion_tokens when the flag is on."""
    from inference_perf.config.reportgen.config import GoodputConfig

    def make_metric() -> RequestLifecycleMetric:
        info = InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=10,  # client count
                output_token_times=[1.0, 3.0],
                server_usage={"completion_tokens": 5},  # server count
            ),
        )
        return RequestLifecycleMetric(
            scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None
        )

    # A lenient latency constraint keeps the single request "good"; benchmark
    # window is 10s, so token_goodput == good_total_tokens / 10.
    goodput_config = GoodputConfig(constraints={"request_latency": 100.0})

    default = summarize_requests([make_metric()], [50], goodput_config=goodput_config)
    assert default.successes["goodput_metrics"]["token_goodput"] == pytest.approx((5.0 + 10.0) / 10.0)

    server = summarize_requests([make_metric()], [50], goodput_config=goodput_config, use_server_output_tokens=True)
    assert server.successes["goodput_metrics"]["token_goodput"] == pytest.approx((5.0 + 5.0) / 10.0)


def test_enrich_sessions_honors_use_server_output_tokens() -> None:
    """Per-session total_output_tokens uses the server count when the flag is on."""
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(output_tokens=10, server_usage={"completion_tokens": 5}),
    )
    request_metric = RequestLifecycleMetric(
        scheduled_time=0.0,
        start_time=0.0,
        end_time=10.0,
        request_data="r",
        info=info,
        error=None,
        session_id="s1",
    )

    def make_session() -> SessionLifecycleMetric:
        return SessionLifecycleMetric(
            session_id="s1",
            stage_id=0,
            file_path="s1.json",
            start_time=0.0,
            end_time=10.0,
            duration_sec=10.0,
            num_events=1,
            num_events_completed=1,
        )

    # _enrich_sessions doesn't use `self`, so call it unbound with a dummy self.
    default_session = make_session()
    ReportGenerator._enrich_sessions(None, [default_session], [request_metric])  # type: ignore[arg-type]
    assert default_session.total_output_tokens == 10

    server_session = make_session()
    ReportGenerator._enrich_sessions(None, [server_session], [request_metric], use_server_output_tokens=True)  # type: ignore[arg-type]
    assert server_session.total_output_tokens == 5


def _cached_request(session_id: str, input_tokens: int, server_usage: dict[str, Any] | None) -> RequestLifecycleMetric:
    """A session request whose response optionally carries a server usage dict."""
    return RequestLifecycleMetric(
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="r",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=input_tokens)),
            response_metrics=StreamedResponseMetrics(output_tokens=1, server_usage=server_usage),
        ),
        error=None,
        session_id=session_id,
    )


def _cache_session(session_id: str = "s1") -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path=f"{session_id}.json",
        start_time=0.0,
        end_time=1.0,
        duration_sec=1.0,
        num_events=1,
        num_events_completed=1,
    )


def test_enrich_sessions_sums_cached_tokens_across_requests() -> None:
    """total_cached_tokens sums prompt_tokens_details.cached_tokens over a session's requests."""
    requests = [
        _cached_request("s1", 20, {"prompt_tokens": 20, "prompt_tokens_details": {"cached_tokens": 0}}),
        _cached_request("s1", 40, {"prompt_tokens": 40, "prompt_tokens_details": {"cached_tokens": 20}}),
        _cached_request("s1", 39, {"prompt_tokens": 39, "prompt_tokens_details": {"cached_tokens": 12}}),
    ]
    session = _cache_session("s1")
    ReportGenerator._enrich_sessions(None, [session], requests)  # type: ignore[arg-type]

    assert session.total_input_tokens == 99
    assert session.total_cached_tokens == 32  # 0 + 20 + 12
    assert session.total_cacheable_input_tokens == 99  # server prompt_tokens: 20 + 40 + 39


def test_enrich_sessions_cached_none_without_server_usage() -> None:
    """No request reporting server usage leaves total_cached_tokens as None (not 0)."""
    requests = [_cached_request("s1", 10, None), _cached_request("s1", 10, None)]
    session = _cache_session("s1")
    ReportGenerator._enrich_sessions(None, [session], requests)  # type: ignore[arg-type]

    assert session.total_cached_tokens is None
    assert session.total_cacheable_input_tokens is None
    assert session.total_input_tokens == 20


def test_enrich_sessions_cached_handles_null_prompt_tokens_details() -> None:
    """server_usage present with prompt_tokens_details: null contributes nothing (see #664)."""
    requests = [
        _cached_request("s1", 10, {"prompt_tokens": 10, "prompt_tokens_details": None}),
        _cached_request("s1", 10, {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}}),
    ]
    session = _cache_session("s1")
    ReportGenerator._enrich_sessions(None, [session], requests)  # type: ignore[arg-type]

    # The null-details request reported no cache info at all, so it contributes to
    # neither side of the ratio — only the request that did report is counted.
    assert session.total_cached_tokens == 4
    assert session.total_cacheable_input_tokens == 10


def test_enrich_sessions_cache_denominator_uses_server_prompt_tokens() -> None:
    """The hit-rate denominator is the server's prompt_tokens, not the client-side count.

    Regression: the denominator was total_input_tokens (a client-side
    re-tokenization). The client undercounts what the server actually prompts on
    -- chat template and tool-schema overhead it does not model -- so pairing it
    with the server-reported cached_tokens inflated the rate, here past 1.0.
    """
    # Client re-tokenizes 100 tokens; the server prompted on 120 and cached 110.
    requests = [_cached_request("s1", 100, {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 110}})]
    session = _cache_session("s1")
    ReportGenerator._enrich_sessions(None, [session], requests)  # type: ignore[arg-type]

    assert session.total_input_tokens == 100  # client count, unchanged
    assert session.total_cacheable_input_tokens == 120  # server count drives the ratio

    summary = ReportGenerator.summarize_sessions(None, [session], DEFAULT_PERCENTILES)  # type: ignore[arg-type]
    assert summary["kv_cache_hit_rate"]["mean"] == pytest.approx(110 / 120)
    assert summary["kv_cache_hit_rate"]["max"] <= 1.0


def test_enrich_sessions_cache_matches_stage_level_prompt_token_usage() -> None:
    """Per-session hit rate agrees with the stage-level prompt_token_usage split.

    Regression: the two lived in one report and disagreed (0.6621 vs 0.64 on this
    data) because only one of them used the server-side denominator.
    """
    client = [900, 1900, 2900, 3900, 4900]  # client under-counts each turn
    server = [1000, 2000, 3000, 4000, 5000]
    cached = [0, 900, 1900, 2900, 3900]
    requests = [
        _cached_request("s1", c, {"prompt_tokens": s, "prompt_tokens_details": {"cached_tokens": k}})
        for c, s, k in zip(client, server, cached, strict=True)
    ]
    session = _cache_session("s1")
    ReportGenerator._enrich_sessions(None, [session], requests)  # type: ignore[arg-type]

    stage = summarize_prompt_token_usage(requests, DEFAULT_PERCENTILES)
    summary = ReportGenerator.summarize_sessions(None, [session], DEFAULT_PERCENTILES)  # type: ignore[arg-type]

    assert summary["kv_cache_hit_rate_aggregate"] == pytest.approx(stage["cached"] / stage["total"])


def test_enrich_sessions_cache_reads_anthropic_usage_shape() -> None:
    """Anthropic's cache_read_input_tokens counts, and its input_tokens excludes it.

    Regression: only the OpenAI/vLLM shape was read, so an Anthropic run reported
    total_cached_tokens=0 -- indistinguishable from a genuine 0% hit rate.
    """
    requests = [_cached_request("s1", 100, {"input_tokens": 20, "output_tokens": 5, "cache_read_input_tokens": 80})]
    session = _cache_session("s1")
    ReportGenerator._enrich_sessions(None, [session], requests)  # type: ignore[arg-type]

    assert session.total_cached_tokens == 80
    # Anthropic's input_tokens is the uncached remainder, so the prompt total is 20 + 80.
    assert session.total_cacheable_input_tokens == 100

    summary = ReportGenerator.summarize_sessions(None, [session], DEFAULT_PERCENTILES)  # type: ignore[arg-type]
    assert summary["kv_cache_hit_rate"]["mean"] == pytest.approx(0.8)


def test_enrich_sessions_cache_clamps_cached_above_prompt_tokens() -> None:
    """A cached count exceeding prompt_tokens is clamped instead of yielding a rate > 1."""
    requests = [_cached_request("s1", 100, {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 150}})]
    session = _cache_session("s1")
    ReportGenerator._enrich_sessions(None, [session], requests)  # type: ignore[arg-type]

    assert session.total_cached_tokens == 100

    summary = ReportGenerator.summarize_sessions(None, [session], DEFAULT_PERCENTILES)  # type: ignore[arg-type]
    assert summary["kv_cache_hit_rate"]["max"] == pytest.approx(1.0)


def test_summarize_sessions_reports_kv_cache_hit_rate() -> None:
    """summarize_sessions emits per-session cached totals and the cached/prompt ratio."""
    with_cache = _cache_session("s1")
    with_cache.total_input_tokens = 99
    with_cache.total_cached_tokens = 32  # golden shape from a real qwen32 run
    with_cache.total_cacheable_input_tokens = 99

    no_cache = _cache_session("s2")
    no_cache.total_input_tokens = 50
    no_cache.total_cached_tokens = None  # no server usage — excluded from the ratio
    no_cache.total_cacheable_input_tokens = None

    # summarize_sessions doesn't use `self`, so call it unbound with a dummy self.
    summary = ReportGenerator.summarize_sessions(None, [with_cache, no_cache], DEFAULT_PERCENTILES)  # type: ignore[arg-type]

    assert summary["total_cached_tokens"]["mean"] == pytest.approx(32.0)
    # Only s1 contributes: 32/99. s2 (None) is skipped, not treated as 0.
    assert summary["kv_cache_hit_rate"]["mean"] == pytest.approx(32.0 / 99.0)
    assert summary["sessions_with_cache_info"] == 1


def test_summarize_sessions_kv_cache_hit_rate_none_when_no_cache_info() -> None:
    """With no session reporting cache info, the ratio summary is None, not a crash."""
    s = _cache_session("s1")
    s.total_input_tokens = 30
    s.total_cached_tokens = None

    summary = ReportGenerator.summarize_sessions(None, [s], DEFAULT_PERCENTILES)  # type: ignore[arg-type]

    assert summary["kv_cache_hit_rate"] is None
    assert summary["kv_cache_hit_rate_aggregate"] is None
    assert summary["total_cached_tokens"] is None
    assert summary["sessions_with_cache_info"] == 0


def test_summarize_sessions_kv_cache_aggregate_is_token_weighted() -> None:
    """The aggregate rate weights sessions by tokens; the percentile mean does not.

    A tiny fully-cached session and a huge uncached one average to 50% as a
    mean-of-ratios, which misreads a run that cached ~0.1% of its tokens.
    """
    small = _cache_session("small")
    small.total_cached_tokens, small.total_cacheable_input_tokens = 10, 10

    large = _cache_session("large")
    large.total_cached_tokens, large.total_cacheable_input_tokens = 0, 10000

    summary = ReportGenerator.summarize_sessions(None, [small, large], DEFAULT_PERCENTILES)  # type: ignore[arg-type]

    assert summary["kv_cache_hit_rate"]["mean"] == pytest.approx(0.5)  # mean of ratios
    assert summary["kv_cache_hit_rate_aggregate"] == pytest.approx(10 / 10010)  # token-weighted


def test_use_server_output_tokens_falls_back_without_usage() -> None:
    """With the flag on but no server usage, normalization uses the client count."""
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(output_tokens=10, output_token_times=[1.0, 3.0]),  # no server_usage
    )
    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None)

    result = summarize_requests([metric], [50], use_server_output_tokens=True)

    assert result.successes["latency"]["time_per_output_token"]["mean"] == pytest.approx(2.0 / 9.0)


def test_summarize_requests_surfaces_prompt_cache_usage() -> None:
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=10)),
        response_metrics=StreamedResponseMetrics(
            output_tokens=2,
            server_usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
        ),
    )

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="test_request", info=info, error=None
    )

    # Single value: every percentile collapses to it.
    result = summarize_requests([metric], DEFAULT_PERCENTILES)

    expected = {"total": 10.0, "cached": 6.0, "uncached": 4.0, "mean": 10.0, "min": 10.0, "max": 10.0}
    for p in DEFAULT_PERCENTILES:
        expected["median" if p == 50 else f"p{p:g}"] = 10.0
    assert result.successes["prompt_tokens"] == pytest.approx(expected)


def test_summarize_requests_handles_null_prompt_tokens_details() -> None:
    """Servers that report `prompt_tokens_details: null` must not crash."""
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=10)),
        response_metrics=UnaryResponseMetrics(
            output_tokens=2,
            server_usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": None,
            },
        ),
    )

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="test_request", info=info, error=None
    )

    result = summarize_requests([metric], DEFAULT_PERCENTILES)

    prompt_tokens = result.successes["prompt_tokens"]
    assert prompt_tokens["total"] == pytest.approx(10.0)
    assert prompt_tokens["cached"] == pytest.approx(0.0)
    assert prompt_tokens["uncached"] == pytest.approx(10.0)
