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
"""Unit tests for the Prometheus collection flows: `execute_query` and `collect_metrics_*`.

`test_prometheus_query_goldens.py` pins the PromQL a metric renders (it drives
`metric.get_queries` directly and never touches this client), `test_prometheus_client.py`
pins how results are assembled into `ModelServerMetrics` with `execute_query` patched
out, and the report-generator tests patch `collect_metrics_summary` /
`collect_metrics_for_stage` out. That left the HTTP round trip and the two collection
entry points with no coverage of their own.

Two behaviours make this mostly a silent-wrongness surface rather than a crash surface:

- every failure path `execute_query` handles returns 0.0, so a non-200, an error body,
  or an unparseable value reaches the report as a real-looking zero rather than an
  error. Some of those zeros leave an error log behind; an empty result set and the
  tolerated malformed shapes do not even do that. The handling has holes, pinned in
  `TestExecuteQueryRaisingPaths`: a 200 whose body is not JSON, a null sample value,
  or a body that is not a JSON object raises out of `execute_query` instead (only
  `raise_for_status` and the float conversion are guarded).
- `collect_metrics_for_stage` derives the query window from the stage's own timestamps
  plus the scrape interval and buffer. Wrong arithmetic there reads the wrong samples
  and reports numbers that belong to another stage.
"""

import math
from typing import Any, Dict, Optional, cast
from unittest.mock import Mock, patch

import pytest
import requests

from inference_perf.client.modelserver.metrics import BaseMetrics, GaugeMetric
from inference_perf.client.server_metrics.base import (
    ModelServerMetrics,
    PerfRuntimeParameters,
    StageRuntimeInfo,
    StageStatus,
)
from inference_perf.client.server_metrics.prometheus_client.base import (
    PROMETHEUS_SCRAPE_BUFFER_SEC,
    PrometheusMetricsClient,
)
from inference_perf.config import PrometheusClientConfig

QUERY_URL = "http://localhost:9090/api/v1/query"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Builds a client pointed at http://localhost:9090 with a 30s scrape interval,
# so query_url is http://localhost:9090/api/v1/query.
def _client(scrape_interval: int = 30) -> PrometheusMetricsClient:
    config = PrometheusClientConfig(url="http://localhost:9090", scrape_interval=scrape_interval)
    return PrometheusMetricsClient(config)


# Fakes one HTTP response. json() returns `payload`; raise_for_status() raises `error` if
# given, otherwise does nothing. Stands in for the requests.Response execute_query gets back.
def _response(payload: Any, error: Optional[Exception] = None) -> Mock:
    response = Mock()
    response.json.return_value = payload
    if error is not None:
        response.raise_for_status.side_effect = error
    else:
        response.raise_for_status.return_value = None
    return response


# The shape Prometheus returns for a successful instant query: one sample whose value is
# the pair [timestamp, "<value as string>"]. `value` becomes that second element.
def _vector(value: str) -> Dict[str, Any]:
    return {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1632741820.781, value]}]},
    }


# Builds runtime parameters with the given stages (default none) and start time (default
# t=0), carrying an empty BaseMetrics so get_model_server_metrics runs no queries unless
# a real `metrics` container is supplied.
def _runtime(
    stages: Optional[Dict[int, StageRuntimeInfo]] = None,
    start_time: float = 0.0,
    metrics: Optional[BaseMetrics] = None,
) -> PerfRuntimeParameters:
    return PerfRuntimeParameters(
        start_time=start_time,
        duration=10.0,
        model_server_metrics=metrics if metrics is not None else BaseMetrics(),
        stages=stages if stages is not None else {},
    )


# One stage running from start_time to end_time, defaulting to 50.0 -> 100.0, COMPLETED
# unless another status is given.
def _stage(
    stage_id: int = 0,
    start_time: float = 50.0,
    end_time: float = 100.0,
    status: StageStatus = StageStatus.COMPLETED,
) -> StageRuntimeInfo:
    return StageRuntimeInfo(
        stage_id=stage_id,
        rate=2.0,
        start_time=start_time,
        end_time=end_time,
        status=status,
    )


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


class TestExecuteQueryHappyPath:
    # A well-formed vector carrying "1.5" returns 1.5.
    def test_parses_the_first_sample_value(self) -> None:
        client = _client()

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(_vector("1.5"))
            result = client.execute_query("up", "100")

        assert result == 1.5

    # Querying 'up' at eval time "100" issues GET http://localhost:9090/api/v1/query with
    # params {"query": "up", "time": "100"} and the client's headers (base returns {}).
    def test_sends_the_query_and_eval_time_to_the_query_url(self) -> None:
        client = _client()

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(_vector("1.0"))
            client.execute_query("up", "100")

        get.assert_called_once_with(QUERY_URL, headers={}, params={"query": "up", "time": "100"})

    # Prometheus returns values as strings with full float precision; 0.1234567891 is
    # rounded to 6 decimal places, so the result is 0.123457.
    def test_rounds_the_value_to_six_decimal_places(self) -> None:
        client = _client()

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(_vector("0.1234567891"))
            result = client.execute_query("up", "100")

        assert result == 0.123457

    # More than one series in the result set: only the first sample is read, so a payload
    # whose samples are "7.0" then "9.0" returns 7.0.
    def test_reads_only_the_first_series(self) -> None:
        client = _client()
        payload = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"pod": "a"}, "value": [1632741820.781, "7.0"]},
                    {"metric": {"pod": "b"}, "value": [1632741820.781, "9.0"]},
                ],
            },
        }

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(payload)
            result = client.execute_query("up", "100")

        assert result == 7.0

    # Prometheus legitimately returns "NaN" (e.g. histogram_quantile over a window with no
    # samples) and "+Inf"; both parse under float(), so they come back as-is rather than
    # as 0.0, and the report then carries non-finite numbers. Documented, not endorsed:
    # pinned so a change in either direction is a conscious one.
    def test_nan_and_inf_pass_through(self) -> None:
        client = _client()

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(_vector("NaN"))
            assert math.isnan(client.execute_query("up", "100"))

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(_vector("+Inf"))
            assert client.execute_query("up", "100") == float("inf")


class TestExecuteQueryReturnsZeroOnFailure:
    """Every one of these reaches the report as a plain 0.0, indistinguishable from a real zero.

    Pinning them documents that the handled failure mode is a silent zero rather than an
    exception, and pins which zeros leave an error log behind: the HTTP and parse
    failures do, an empty result set and the tolerated malformed shapes do not. If a
    future change makes any of these raise or return None instead, these tests are the
    ones to update.
    """

    # A query that matches no series returns an empty result list, which yields 0.0 with
    # no error record at all: an absent metric is indistinguishable from a real zero even
    # in the logs.
    def test_empty_result_set(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()
        payload: Dict[str, Any] = {"status": "success", "data": {"resultType": "vector", "result": []}}

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.return_value = _response(payload)
                assert client.execute_query("up", "100") == 0.0

        assert not [record for record in caplog.records if record.levelname == "ERROR"]

    # A 500 from Prometheus makes raise_for_status raise, which is caught, logged, and
    # yields 0.0. json() must never be reached on this path.
    def test_non_200_response(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()
        response = _response(_vector("1.5"), error=requests.HTTPError("500 Server Error"))

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.return_value = response
                assert client.execute_query("up", "100") == 0.0

        response.json.assert_not_called()
        assert "error executing query: 500 Server Error" in caplog.text

    # A 400 carrying Prometheus' own error body ("bad_data") is caught by raise_for_status
    # before the body is inspected: json() is never called, same as the 500 path. The
    # error body is here to pin that it offers no second route to a result.
    def test_bad_request_with_error_body(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()
        payload = {"status": "error", "errorType": "bad_data", "error": "invalid parameter"}
        response = _response(payload, error=requests.HTTPError("400 Client Error"))

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.return_value = response
                assert client.execute_query("up", "100") == 0.0

        response.json.assert_not_called()
        assert "error executing query: 400 Client Error" in caplog.text

    # A 200 whose body reports status "error" rather than "success" is logged and yields
    # 0.0. The payload carries a well-formed result of "7.0" so this test fails loudly
    # (by returning 7.0) if the status guard is ever removed.
    def test_two_hundred_with_error_status_in_body(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()
        payload = {
            "status": "error",
            "errorType": "bad_data",
            "error": "invalid parameter",
            "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1632741820.781, "7.0"]}]},
        }

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.return_value = _response(payload)
                assert client.execute_query("up", "100") == 0.0

        assert "error executing query" in caplog.text
        assert "bad_data" in caplog.text

    # The connection never gets made (server down, DNS failure). The exception is caught,
    # logged, and yields 0.0 rather than propagating out of report generation.
    def test_connection_error(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.side_effect = requests.ConnectionError("connection refused")
                assert client.execute_query("up", "100") == 0.0

        assert "error executing query: connection refused" in caplog.text

    # A sample value float() cannot parse at all fails the conversion, is logged, and
    # yields 0.0. Real "NaN"/"+Inf" strings DO parse and are not this case; they pass
    # through unconverted (see test_nan_and_inf_pass_through).
    def test_non_numeric_value(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.return_value = _response(_vector("not-a-number"))
                assert client.execute_query("up", "100") == 0.0

        assert "error converting value to float: not-a-number" in caplog.text

    # The defensive branch for a None response object (requests.get does not do this, but
    # the code guards for it) yields 0.0. The guard's own log line names the query ("up"),
    # unlike the exception path's, so the assertion fails if the guard is removed and the
    # AttributeError from None.raise_for_status() is handled instead.
    def test_none_response(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.return_value = None
                assert client.execute_query("up", "100") == 0.0

        assert "error executing query: up" in caplog.text

    # Malformed payloads that fall out of the dict/list checks yield 0.0, with no error
    # record at all: a sample with no "value" key, a scalar "value", a one-element
    # "value", and a body with no "data" key. This list is curated, not exhaustive;
    # malformed shapes that raise instead are pinned in TestExecuteQueryRaisingPaths.
    @pytest.mark.parametrize(
        "payload",
        [
            {"status": "success", "data": {"result": [{"metric": {}}]}},
            {"status": "success", "data": {"result": [{"metric": {}, "value": "1.5"}]}},
            {"status": "success", "data": {"result": [{"metric": {}, "value": [1632741820.781]}]}},
            {"status": "success"},
        ],
        ids=["no-value-key", "scalar-value", "one-element-value", "no-data-key"],
    )
    def test_malformed_result_shapes(self, payload: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        client = _client()

        with caplog.at_level("ERROR"):
            with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
                get.return_value = _response(payload)
                assert client.execute_query("up", "100") == 0.0

        assert not [record for record in caplog.records if record.levelname == "ERROR"]


class TestExecuteQueryRaisingPaths:
    """Not every bad response is absorbed into 0.0: these escape `execute_query`.

    `response.json()` sits outside the try/except and only ValueError is caught around
    the float conversion, so each of these propagates out of report generation and would
    kill the report of a completed run. Pinned as documentation of the current contract,
    not as endorsement; if the handling is ever widened to absorb them, these are the
    tests to flip.
    """

    # A 200 whose body is not JSON (a proxy or ingress answering with an HTML error
    # page): response.json() raises requests' JSONDecodeError, which escapes.
    def test_non_json_body_raises(self) -> None:
        client = _client()
        response = _response(None)
        response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = response
            with pytest.raises(requests.exceptions.JSONDecodeError):
                client.execute_query("up", "100")

    # A well-formed sample whose value is JSON null: float(None) raises TypeError, which
    # the ValueError-only handler around the conversion lets escape.
    def test_null_sample_value_raises(self) -> None:
        client = _client()
        payload = {
            "status": "success",
            "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1632741820.781, None]}]},
        }

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(payload)
            with pytest.raises(TypeError):
                client.execute_query("up", "100")

    # A body that is JSON but not an object (a bare array): .get does not exist on a
    # list, so AttributeError escapes before any status check.
    def test_array_body_raises(self) -> None:
        client = _client()

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response([])
            with pytest.raises(AttributeError):
                client.execute_query("up", "100")


# ---------------------------------------------------------------------------
# collect_metrics_summary
# ---------------------------------------------------------------------------


class TestCollectMetricsSummary:
    # No runtime parameters at all (collection ran before the benchmark registered them):
    # returns None and logs a warning rather than raising.
    def test_returns_none_without_runtime_parameters(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()

        with caplog.at_level("WARNING"):
            assert client.collect_metrics_summary(cast(PerfRuntimeParameters, None)) is None

        assert "Perf Runtime parameters are not set" in caplog.text

    # A run that started at t=100 collected at wall-clock t=500: the eval time is 500.0
    # and the duration is 400.0 (500 - 100). The two values are distinct so this also
    # pins the (metrics, duration, eval_time) argument order. Note the duration is
    # wall-clock time since the run started, including the post-run scrape wait, so
    # per-second summary metrics divide by more than the run itself; pinned as
    # documented behaviour.
    def test_window_spans_from_run_start_to_now(self) -> None:
        client = _client()
        runtime = _runtime(start_time=100.0)

        with (
            patch("inference_perf.client.server_metrics.prometheus_client.base.time.time", return_value=500.0),
            patch.object(PrometheusMetricsClient, "get_model_server_metrics") as get_metrics,
        ):
            client.collect_metrics_summary(runtime)

        get_metrics.assert_called_once_with(runtime.model_server_metrics, 400.0, 500.0)

    # The ModelServerMetrics built from the queries is passed straight back to the caller.
    def test_returns_the_assembled_metrics(self) -> None:
        client = _client()
        metrics = ModelServerMetrics()

        with (
            patch("inference_perf.client.server_metrics.prometheus_client.base.time.time", return_value=500.0),
            patch.object(PrometheusMetricsClient, "get_model_server_metrics", return_value=metrics),
        ):
            assert client.collect_metrics_summary(_runtime()) is metrics


# ---------------------------------------------------------------------------
# collect_metrics_for_stage
# ---------------------------------------------------------------------------


class TestCollectMetricsForStage:
    # No runtime parameters: returns None and warns, same as the summary path.
    def test_returns_none_without_runtime_parameters(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()

        with caplog.at_level("WARNING"):
            assert client.collect_metrics_for_stage(cast(PerfRuntimeParameters, None), 0) is None

        assert "Perf Runtime parameters are not set" in caplog.text

    # Asking for stage 1 when only stage 0 ran: returns None and names the missing stage,
    # so the report generator skips that stage instead of publishing another stage's numbers.
    def test_returns_none_for_an_unknown_stage(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()
        runtime = _runtime({0: _stage(0)})

        with caplog.at_level("WARNING"):
            assert client.collect_metrics_for_stage(runtime, 1) is None

        assert "Stage ID 1 is not present" in caplog.text

    # Runtime parameters carrying no stages map at all: returns None and warns.
    def test_returns_none_when_stages_are_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client()
        runtime = _runtime()
        runtime.stages = cast(Dict[int, StageRuntimeInfo], None)

        with caplog.at_level("WARNING"):
            assert client.collect_metrics_for_stage(runtime, 0) is None

        assert "Stage ID 0 is not present" in caplog.text

    # The window arithmetic, pinned as documented behaviour. Stage 0 ran 50.0 -> 100.0
    # with a 30s scrape interval, so the eval time is 100 + 30 + 2 = 132.0 (the 2 is
    # PROMETHEUS_SCRAPE_BUFFER_SEC, asserted below so the literal arithmetic in these
    # comments stays honest) and the duration is 132 - 50 = 82.0. Evaluating past
    # end_time is what scrape lag requires, but folding the extra 32s into the duration
    # widens the PromQL range the query builders embed, so rate()-style metrics for this
    # 50s stage divide by 82s. That dilution is documented here, not endorsed; making
    # the window tight is a behaviour change that should turn exactly this test red.
    def test_window_extends_past_stage_end_by_the_scrape_interval_and_buffer(self) -> None:
        client = _client(scrape_interval=30)
        runtime = _runtime({0: _stage(0, start_time=50.0, end_time=100.0)})

        with patch.object(PrometheusMetricsClient, "get_model_server_metrics") as get_metrics:
            client.collect_metrics_for_stage(runtime, 0)

        assert PROMETHEUS_SCRAPE_BUFFER_SEC == 2
        get_metrics.assert_called_once_with(runtime.model_server_metrics, 82.0, 132.0)

    # A shorter 5s scrape interval moves the window with it: stage 0 running 50.0 -> 100.0
    # gives an eval time of 100 + 5 + 2 = 107.0 and a duration of 107 - 50 = 57.0.
    def test_window_follows_the_configured_scrape_interval(self) -> None:
        client = _client(scrape_interval=5)
        runtime = _runtime({0: _stage(0, start_time=50.0, end_time=100.0)})

        with patch.object(PrometheusMetricsClient, "get_model_server_metrics") as get_metrics:
            client.collect_metrics_for_stage(runtime, 0)

        get_metrics.assert_called_once_with(runtime.model_server_metrics, 57.0, 107.0)

    # Stages far enough apart get windows derived from their own timestamps that do not
    # overlap. Stage 0 ran 0 -> 10 and stage 1 ran 50 -> 60 at a 30s interval, so stage 0
    # evaluates at 10 + 32 = 42.0 over 42.0 (queried range [0, 42]) and stage 1 at
    # 60 + 32 = 92.0 over 42.0 (queried range [50, 92]).
    def test_each_stage_gets_its_own_window(self) -> None:
        client = _client(scrape_interval=30)
        runtime = _runtime(
            {
                0: _stage(0, start_time=0.0, end_time=10.0),
                1: _stage(1, start_time=50.0, end_time=60.0),
            }
        )

        with patch.object(PrometheusMetricsClient, "get_model_server_metrics") as get_metrics:
            client.collect_metrics_for_stage(runtime, 0)
            client.collect_metrics_for_stage(runtime, 1)

        windows = [(call.args[1], call.args[2]) for call in get_metrics.call_args_list]
        assert windows == [(42.0, 42.0), (42.0, 92.0)]

    # The isolation above needs that gap: a successor starting within scrape_interval +
    # buffer of a stage's end lands inside that stage's queried range. Stage 0 ran 0 -> 10
    # and stage 1 ran 20 -> 30, so stage 0's range [0, 42] contains all of stage 1's run
    # and stage 0's rate/increase queries count stage 1's traffic too. Documented, not
    # endorsed: this is the cross-stage contamination the module docstring warns about.
    def test_close_stages_query_overlapping_windows(self) -> None:
        client = _client(scrape_interval=30)
        runtime = _runtime(
            {
                0: _stage(0, start_time=0.0, end_time=10.0),
                1: _stage(1, start_time=20.0, end_time=30.0),
            }
        )

        with patch.object(PrometheusMetricsClient, "get_model_server_metrics") as get_metrics:
            client.collect_metrics_for_stage(runtime, 0)
            client.collect_metrics_for_stage(runtime, 1)

        windows = [(call.args[1], call.args[2]) for call in get_metrics.call_args_list]
        assert windows == [(42.0, 42.0), (42.0, 62.0)]
        # Stage 0's queried range is [eval - duration, eval] = [0.0, 42.0]: it contains
        # stage 1's whole run (20.0 -> 30.0).
        stage0_range_start, stage0_range_end = windows[0][1] - windows[0][0], windows[0][1]
        assert stage0_range_start <= 20.0 and 30.0 <= stage0_range_end

    # Stage status is ignored: a FAILED stage is queried exactly like a COMPLETED one
    # over the same window, and the report generator calls this for every stage
    # regardless of status, so a failed stage's numbers are published looking real.
    # Documented, not endorsed; if a status guard is ever added, this is the test that
    # should flip.
    def test_failed_stage_is_still_queried(self) -> None:
        client = _client(scrape_interval=30)
        runtime = _runtime({0: _stage(0, start_time=50.0, end_time=100.0, status=StageStatus.FAILED)})

        with patch.object(PrometheusMetricsClient, "get_model_server_metrics") as get_metrics:
            client.collect_metrics_for_stage(runtime, 0)

        get_metrics.assert_called_once_with(runtime.model_server_metrics, 82.0, 132.0)

    # The ModelServerMetrics built for the stage is passed straight back to the caller.
    def test_returns_the_assembled_metrics(self) -> None:
        client = _client()
        metrics = ModelServerMetrics()
        runtime = _runtime({0: _stage(0)})

        with patch.object(PrometheusMetricsClient, "get_model_server_metrics", return_value=metrics):
            assert client.collect_metrics_for_stage(runtime, 0) is metrics


# ---------------------------------------------------------------------------
# The full collection path, nothing in between patched
# ---------------------------------------------------------------------------


class TestCollectionThroughTheWire:
    # Drives collect_metrics_for_stage through get_model_server_metrics and execute_query
    # down to requests.get with only the HTTP layer faked: one real GaugeMetric issues its
    # four quantile queries, every GET carries the stage's eval time serialized as the
    # float string "132.0" (time=str(query_eval_time)) with the 82s duration embedded in
    # the PromQL range, and the "1.5" the fake server answers comes back as
    # queue_length.avg. This is the only test that reaches the float-to-string seam; int
    # truncation or moving the str() call changes what is actually sent to Prometheus.
    def test_stage_collection_sends_the_eval_time_and_parses_the_answer(self) -> None:
        client = _client(scrape_interval=30)
        metrics = BaseMetrics(custom_metrics={"queue_length": GaugeMetric("vllm:num_requests_waiting")})
        runtime = _runtime({0: _stage(0, start_time=50.0, end_time=100.0)}, metrics=metrics)

        with patch("inference_perf.client.server_metrics.prometheus_client.base.requests.get") as get:
            get.return_value = _response(_vector("1.5"))
            result = client.collect_metrics_for_stage(runtime, 0)

        assert get.call_count == 4
        assert [call.kwargs["params"]["time"] for call in get.call_args_list] == ["132.0"] * 4
        assert "[82s]" in get.call_args_list[0].kwargs["params"]["query"]
        assert result is not None
        assert result.queue_length.avg == 1.5


# ---------------------------------------------------------------------------
# Construction and scrape wait
# ---------------------------------------------------------------------------


class TestClientConstruction:
    # A url with a trailing slash still yields exactly one /api/v1/query suffix.
    def test_normalizes_the_query_url(self) -> None:
        client = PrometheusMetricsClient(PrometheusClientConfig(url="http://localhost:9090/"))

        assert client.query_url == QUERY_URL

    # An unset scrape interval takes PrometheusClientConfig's own default of 15, not the 30
    # in `config.scrape_interval or 30`. That 30 is reachable only for a falsy value, so a
    # config that sets scrape_interval to 0 silently gets 30 seconds instead.
    def test_scrape_interval_defaults_to_the_config_default(self) -> None:
        unset = PrometheusMetricsClient(PrometheusClientConfig(url="http://localhost:9090"))
        zero = PrometheusMetricsClient(PrometheusClientConfig(url="http://localhost:9090", scrape_interval=0))

        assert unset.scrape_interval == 15
        assert zero.scrape_interval == 30

    # No config at all: constructing raises rather than producing a client that would
    # query nothing and report zeros.
    def test_rejects_a_missing_config(self) -> None:
        with pytest.raises(Exception, match="prometheus config missing"):
            PrometheusMetricsClient(cast(PrometheusClientConfig, None))

    # A config whose url is unset raises for the same reason.
    def test_rejects_a_missing_url(self) -> None:
        config = PrometheusClientConfig(url="http://localhost:9090")
        config.url = cast(Any, None)

        with pytest.raises(Exception, match="prometheus url missing"):
            PrometheusMetricsClient(config)


class TestWait:
    # wait() sleeps one full scrape interval plus the buffer, so the last requests of a
    # stage are scraped before they are queried: 30 + 2 = 32 seconds.
    def test_sleeps_for_the_scrape_interval_plus_buffer(self) -> None:
        client = _client(scrape_interval=30)

        with patch("inference_perf.client.server_metrics.prometheus_client.base.time.sleep") as sleep:
            client.wait()

        sleep.assert_called_once_with(32)

    # A 5s scrape interval sleeps 5 + 2 = 7 seconds.
    def test_follows_the_configured_scrape_interval(self) -> None:
        client = _client(scrape_interval=5)

        with patch("inference_perf.client.server_metrics.prometheus_client.base.time.sleep") as sleep:
            client.wait()

        sleep.assert_called_once_with(7)
