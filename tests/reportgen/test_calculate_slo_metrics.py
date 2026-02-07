import pytest
from unittest.mock import Mock
from inference_perf.apis import RequestLifecycleMetric
from inference_perf.reportgen.base import calculate_slo_metrics

@pytest.fixture(scope="module", autouse=True)
def test_calculate_slo_metrics_empty_metrics() -> None:
    """Test with empty metrics list returns None."""
    result = calculate_slo_metrics([], [], [])
    assert result is None



def test_calculate_slo_metrics_no_slo_thresholds() -> None:
    """Test with metrics but no SLO thresholds returns None."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = None
    metric.tpot_slo_sec = None
    metric.start_time = 0.0
    metric.end_time = 1.0
    
    result = calculate_slo_metrics([metric], [None], [None])
    assert result is None



def test_calculate_slo_metrics_ttft_slo_met() -> None:
    """Test TTFT SLO calculation when SLO is met."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = 0.5
    metric.tpot_slo_sec = None
    metric.start_time = 0.0
    metric.end_time = 2.0
    metric.info = Mock()
    metric.info.input_tokens = 10
    metric.info.output_tokens = 5
    
    ttft_value = 0.3  # Below SLO
    result = calculate_slo_metrics([metric], [ttft_value], [None])
    
    assert result is not None
    assert "ttft_slo" in result
    assert result["ttft_slo"]["attainment_pct"] == 100.0
    assert result["ttft_slo"]["requests_met"] == 1
    assert result["ttft_slo"]["requests_failed"] == 0
    assert result["ttft_slo"]["slo"] == 0.5



def test_calculate_slo_metrics_ttft_slo_failed() -> None:
    """Test TTFT SLO calculation when SLO is not met."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = 0.2
    metric.tpot_slo_sec = None
    metric.start_time = 0.0
    metric.end_time = 2.0
    metric.info = Mock()
    metric.info.input_tokens = 10
    metric.info.output_tokens = 5
    
    ttft_value = 0.5  # Above SLO
    result = calculate_slo_metrics([metric], [ttft_value], [None])
    
    assert result is not None
    assert result["ttft_slo"]["attainment_pct"] == 0.0
    assert result["ttft_slo"]["requests_met"] == 0
    assert result["ttft_slo"]["requests_failed"] == 1



def test_calculate_slo_metrics_ttft_slo_missing_value() -> None:
    """Test TTFT SLO when ttft value is None (non-streamable)."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = 0.5
    metric.tpot_slo_sec = None
    metric.start_time = 0.0
    metric.end_time = 2.0
    metric.info = Mock()
    metric.info.input_tokens = 10
    metric.info.output_tokens = 5
    
    result = calculate_slo_metrics([metric], [None], [None])
    
    assert result is not None
    assert result["ttft_slo"]["attainment_pct"] == 100.0
    assert result["ttft_slo"]["requests_failed"] == 0



def test_calculate_slo_metrics_tpot_slo_met() -> None:
    """Test TPOT SLO calculation when SLO is met."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = None
    metric.tpot_slo_sec = 0.1
    metric.start_time = 0.0
    metric.end_time = 2.0
    metric.info = Mock()
    metric.info.input_tokens = 10
    metric.info.output_tokens = 5
    
    tpot_value = 0.08  # Below SLO
    result = calculate_slo_metrics([metric], [None], [tpot_value])
    
    assert result is not None
    assert "tpot_slo" in result
    assert result["tpot_slo"]["attainment_pct"] == 100.0
    assert result["tpot_slo"]["requests_met"] == 1
    assert result["tpot_slo"]["slo"] == 0.1



def test_calculate_slo_metrics_combined_slo() -> None:
    """Test combined SLO when both TTFT and TPOT are present."""
    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.ttft_slo_sec = 0.5
    metric1.tpot_slo_sec = 0.1
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.input_tokens = 10
    metric1.info.output_tokens = 5
    
    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.ttft_slo_sec = 0.5
    metric2.tpot_slo_sec = 0.1
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.input_tokens = 10
    metric2.info.output_tokens = 5
    
    ttft_values: list[float | None] = [0.3, 0.6]
    tpot_values: list[float | None] = [0.08, 0.08]
    
    result = calculate_slo_metrics(
        [metric1, metric2],
        ttft_values,
        tpot_values
    )
    
    assert result is not None
    assert "combined_slo" in result
    assert result["combined_slo"]["requests_met"] == 1  # Only metric1 meets both
    assert result["combined_slo"]["requests_failed"] == 1



def test_calculate_slo_metrics_multiple_requests() -> None:
    """Test SLO calculation with multiple requests."""
    metrics: list[RequestLifecycleMetric] = []
    ttft_values: list[float | None] = []
    tpot_values: list[float | None] = []
    
    for i in range(5):
        metric = Mock(spec=RequestLifecycleMetric)
        metric.ttft_slo_sec = 0.5
        metric.tpot_slo_sec = None
        metric.start_time = float(i)
        metric.end_time = float(i + 1)
        metric.info = Mock()
        metric.info.input_tokens = 10
        metric.info.output_tokens = 5
        metrics.append(metric)
        ttft_values.append(0.3 if i % 2 == 0 else 0.6)  # 60% pass
        tpot_values.append(None)
    
    result = calculate_slo_metrics(metrics, ttft_values, tpot_values)
    
    assert result is not None
    assert result["ttft_slo"]["attainment_pct"] == 60.0
    assert result["ttft_slo"]["requests_met"] == 3
    assert result["ttft_slo"]["requests_failed"] == 2



def test_calculate_slo_metrics_goodput_calculation() -> None:
    """Test goodput calculation in combined SLO metrics."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = 0.5
    metric.tpot_slo_sec = 0.1
    metric.start_time = 0.0
    metric.end_time = 10.0
    metric.info = Mock()
    metric.info.input_tokens = 100
    metric.info.output_tokens = 50
    
    result = calculate_slo_metrics([metric], [0.3], [0.08])
    
    assert result is not None
    assert "combined_slo" in result
    assert "goodput_rate" in result["combined_slo"]
    assert result["combined_slo"]["goodput_rate"] == 15.0  # (100+50) / 10