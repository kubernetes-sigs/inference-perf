# Copyright 2025 The Kubernetes Authors.
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
from unittest.mock import MagicMock, patch
from inference_perf.client.metricsclient.prometheus_client.base import PrometheusMetricsClient
from inference_perf.config import PrometheusClientConfig
from inference_perf.client.modelserver.base import ModelServerPrometheusMetric
from pydantic import HttpUrl

def test_collect_raw_metrics() -> None:
    config = PrometheusClientConfig(
        url=HttpUrl("http://localhost:9090"),
        scrape_interval=15,
        filters=["job='vllm'", "namespace='default'"]
    )
    client = PrometheusMetricsClient(config)

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "metric_name{job=\"vllm\",namespace=\"default\"} 1.0\nmetric_name2{job=\"vllm\"} 2.0"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        raw_metrics = client.collect_raw_metrics(config.filters)

        assert raw_metrics == {"metric_name": "metric_name{job=\"vllm\",namespace=\"default\"} 1.0\n", "metric_name2": "metric_name2{job=\"vllm\"} 2.0\n"}
        mock_get.assert_called_once_with(
            "http://localhost:9090/federate",
            headers={},
            params={"match[]": "{job='vllm',namespace='default'}"}
        )

def test_collect_raw_metrics_error() -> None:
    config = PrometheusClientConfig(
        url=HttpUrl("http://localhost:9090"),
        scrape_interval=15
    )
    client = PrometheusMetricsClient(config)
    
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")
        
        raw_metrics = client.collect_raw_metrics([])
        
        assert raw_metrics is None

def test_collect_raw_metrics_range_query() -> None:
    config = PrometheusClientConfig(
        url=HttpUrl("http://localhost:9090"),
        scrape_interval=15,
        filters=["job='vllm'"]
    )
    client = PrometheusMetricsClient(config)
    
    start_time = 1000.0
    end_time = 1100.0
    interval = 10

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Mocking JSON response for range query
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "vllm_gpu_cache_usage", "instance": "pod1"},
                        "values": [
                            [1000.0, "0.1"],
                            [1010.0, "0.2"]
                        ]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        raw_metrics = client.collect_raw_metrics(
            config.filters, 
            start_time=start_time, 
            end_time=end_time, 
            interval=interval
        )

        assert raw_metrics is not None
        assert "vllm_gpu_cache_usage" in raw_metrics
        # Expected Prometheus text format for range query results (including timestamps in ms)
        expected_line1 = 'vllm_gpu_cache_usage{instance="pod1"} 0.1 1000000'
        expected_line2 = 'vllm_gpu_cache_usage{instance="pod1"} 0.2 1010000'
        assert expected_line1 in raw_metrics["vllm_gpu_cache_usage"]
        assert expected_line2 in raw_metrics["vllm_gpu_cache_usage"]
        
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "http://localhost:9090/api/v1/query_range"
        assert kwargs["params"]["start"] == "1000.0"
        assert kwargs["params"]["end"] == "1100.0"
        assert kwargs["params"]["step"] == "10s"
        assert kwargs["params"]["query"] == "{job='vllm'}"

def test_collect_raw_metrics_google_managed_fallback() -> None:
    # URL containing monitoring.googleapis.com triggers is_google_managed
    config = PrometheusClientConfig(
        url=HttpUrl("https://monitoring.googleapis.com/v1/projects/my-project/locations/global/prometheus"),
        scrape_interval=15,
        filters=["job='vllm'"]
    )
    client = PrometheusMetricsClient(config)

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "vllm_throughput", "instance": "pod1"},
                        "value": [1600000000.0, "15.5"]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        raw_metrics = client.collect_raw_metrics(config.filters)

        assert raw_metrics is not None
        assert "vllm_throughput" in raw_metrics
        assert 'vllm_throughput{instance="pod1"} 15.5' in raw_metrics["vllm_throughput"]
        
        # Should NOT call /federate
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "/api/v1/query" in args[0]
        assert "/federate" not in args[0]

def test_collect_raw_metrics_gmp_with_metadata() -> None:
    config = PrometheusClientConfig(
        url=HttpUrl("https://monitoring.googleapis.com/v1/projects/my-project/locations/global/prometheus"),
        scrape_interval=15,
        filters=["namespace='default'"]
    )
    client = PrometheusMetricsClient(config)
    
    metrics_metadata = {
        "throughput": ModelServerPrometheusMetric(name="vllm_throughput", op="rate", type="counter", filters=["namespace='default'"]),
        "latency": ModelServerPrometheusMetric(name="vllm_latency", op="mean", type="gauge", filters=["namespace='default'"])
    }

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {"resultType": "vector", "result": []}
        }
        mock_get.return_value = mock_response

        client.collect_raw_metrics(config.filters, metrics_metadata=metrics_metadata)

        # GMP with metadata should try individual queries for each metric name
        # 1. general {namespace='default'}
        # 2. vllm_throughput{namespace='default'}
        # 3. vllm_latency{namespace='default'}
        assert mock_get.call_count == 3
        
        queries_sent = [call.kwargs["params"]["query"] for call in mock_get.call_args_list]
        assert "{namespace='default'}" in queries_sent
        assert "vllm_throughput{namespace='default'}" in queries_sent
        assert "vllm_latency{namespace='default'}" in queries_sent
