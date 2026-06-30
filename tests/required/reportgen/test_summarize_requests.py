import pytest
from inference_perf.reportgen.base import summarize_requests
from inference_perf.apis.base import RequestLifecycleMetric, InferenceInfo, StreamedResponseMetrics
from inference_perf.payloads import RequestMetrics, Text


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
    chunk_texts = [hf.decode([i]) for i in ids]
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
    for completion_tokens in (3, 7):
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

    result = summarize_requests(metrics, [50])

    assert result.successes["output_tokens"] == {"total": 10.0}


def test_summarize_requests_output_tokens_falls_back_to_client_count() -> None:
    """Without server usage, output_tokens.total falls back to the client count."""
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=10)),
        response_metrics=StreamedResponseMetrics(output_tokens=4),  # no server_usage
    )
    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="r", info=info, error=None)

    result = summarize_requests([metric], [50])

    assert result.successes["output_tokens"] == {"total": 4.0}


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
    # client count, output_tokens is the server count.
    assert server.successes["output_len"]["mean"] == pytest.approx(10.0)
    assert server.successes["output_tokens"] == {"total": 5.0}


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

    result = summarize_requests([metric], [50])

    assert result.successes["prompt_tokens"] == {
        "total": 10.0,
        "cached": 6.0,
        "uncached": 4.0,
    }
