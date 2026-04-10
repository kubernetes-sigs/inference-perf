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
from unittest.mock import Mock
from inference_perf.apis import RequestLifecycleMetric
from inference_perf.config import GoodputConfig
from inference_perf.reportgen.base import calculate_goodput_metrics


def test_calculate_goodput_metrics_empty_metrics() -> None:
    """Test with empty metrics list returns None."""
    config = GoodputConfig(constraints={"ttft": 0.5})
    result = calculate_goodput_metrics([], config, [], [], [], [], [])
    assert result is None


def test_calculate_goodput_metrics_no_config() -> None:
    """Test with no goodput config returns None."""
    metric = Mock(spec=RequestLifecycleMetric)
    result = calculate_goodput_metrics([metric], None, [0.3], [None], [0.1], [1.0], [None])
    assert result is None


def test_calculate_goodput_metrics_no_constraints() -> None:
    """Test with empty constraints returns None."""
    config = GoodputConfig(constraints={})
    metric = Mock(spec=RequestLifecycleMetric)
    result = calculate_goodput_metrics([metric], config, [0.3], [None], [0.1], [1.0], [None])
    assert result is None


def test_calculate_goodput_metrics_ttft_constraint() -> None:
    """Test goodput calculation with TTFT constraint."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.output_tokens = 10

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.output_tokens = 20

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.6]  # metric1 meets, metric2 fails

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None, None], [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_pct"] == 50.0
    assert result["good_requests"] == 1
    assert result["total_requests"] == 2
    # Total benchmark time = 4.0 - 0.0 = 4.0
    assert result["request_goodput"] == 1 / 4.0
    # Good tokens = 10 (from metric1)
    assert result["token_goodput"] == 10 / 4.0


def test_calculate_goodput_metrics_total_tokens() -> None:
    """Test that token goodput uses total tokens (input + output)."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.input_tokens = 5
    metric1.info.output_tokens = 10

    metrics = [metric1]
    ttft_values: list[float | None] = [0.3]  # Meets constraint

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_pct"] == 100.0
    assert result["good_requests"] == 1
    # Total tokens = 5 + 10 = 15
    # Total benchmark time = 2.0 - 0.0 = 2.0
    assert result["token_goodput"] == 15 / 2.0


def test_calculate_goodput_metrics_multiple_constraints() -> None:
    """Test goodput calculation with multiple constraints."""
    config = GoodputConfig(constraints={"ttft": 0.5, "tpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.output_tokens = 10

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.output_tokens = 20

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.3]  # Both meet TTFT
    tpot_values: list[float | None] = [0.08, 0.12]  # metric1 meets, metric2 fails TPOT

    result = calculate_goodput_metrics(metrics, config, ttft_values, tpot_values, [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_pct"] == 50.0
    assert result["good_requests"] == 1


def test_calculate_goodput_metrics_itl_constraint() -> None:
    """Test goodput calculation with ITL constraint."""
    config = GoodputConfig(constraints={"itl": 0.05})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.output_tokens = 10

    metrics = [metric1]
    itl_values: list[float | None] = [0.04]  # Meets ITL

    result = calculate_goodput_metrics(metrics, config, [None], [None], [0.1], [2.0], itl_values)  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_pct"] == 100.0


def test_calculate_goodput_metrics_ntpot_constraint() -> None:
    """Test goodput calculation with NTPOT constraint."""
    config = GoodputConfig(constraints={"ntpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.output_tokens = 10

    metrics = [metric1]
    ntpot_values: list[float | None] = [0.05]  # Meets NTPOT

    result = calculate_goodput_metrics(metrics, config, [None], [None], ntpot_values, [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_pct"] == 100.0


def test_calculate_goodput_metrics_request_latency_constraint() -> None:
    """Test goodput calculation with request_latency constraint."""
    config = GoodputConfig(constraints={"request_latency": 1.0})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 0.5
    metric1.info = Mock()
    metric1.info.output_tokens = 10

    metrics = [metric1]
    request_latency_values: list[float | None] = [0.5]  # Meets latency

    result = calculate_goodput_metrics(metrics, config, [None], [None], [0.1], request_latency_values, [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_pct"] == 100.0
