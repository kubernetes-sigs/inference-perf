from typing import Any, List, Optional

import requests
from inference_perf.client.client_interfaces.prometheus.base import PrometheusMetricsCollector
from inference_perf.client.client_interfaces.prometheus.prometheus_metrics import PrometheusMetric
from inference_perf.config import GMPCollectorConfig
import google.auth
import google.auth.transport.requests


class GMPMetricsCollector(PrometheusMetricsCollector):
    config: GMPCollectorConfig

    def __init__(self, metrics: List[PrometheusMetric], config: GMPCollectorConfig):
        self.metrics = metrics
        self.config = config
        credentials, project_id = google.auth.default()  # type: ignore[no-untyped-call]
        self.credentials = credentials
        self.project_id = project_id
        self.url = "https://monitoring.googleapis.com/v1/projects/%s/location/global/prometheus/api/v1/metadata" % (
            self.project_id
        )
        print("Created Google Managed Prometheus Metrics Collector")

    async def query_metric(self, query: str, duration: float) -> Optional[float]:
        auth_req = google.auth.transport.requests.Request()  # type: ignore[no-untyped-call]
        self.credentials.refresh(auth_req)

        headers_api = {"Authorization": "Bearer " + self.credentials.token}
        params = {"query": query}
        request_post = requests.get(url=self.url, headers=headers_api)
        all_metrics_metadata = request_post.json()
        if request_post.ok is not True:
            print("HTTP Error: %s" % (all_metrics_metadata))
            return None
        if all_metrics_metadata["status"] != "success":
            print("Metadata error response: %s" % all_metrics_metadata["error"])
            return None

        print(f"Evaluating query: {query}")
        request_post = requests.get(url=self.url, headers=headers_api, params=params)
        response = request_post.json()

        print(f"Got response from metrics server: {response}")
        if request_post.ok:
            if response["status"] == "success" and response["data"] and response["data"]["result"]:
                r = response["data"]["result"]
                if not r:
                    print(f"Failed to get result for {query}")
                    return None
                v = r[0].get("value", None)
                if not v:
                    print(f"Failed to get value for result: {r}")
                    return None
                print(f"Result for {query}: {v[1]}")
                return float(v[1])
            else:
                print("Cloud Monitoring PromQL Error: %s" % (response))
                return None
        else:
            print("HTTP Error: %s" % (response))
            return None
