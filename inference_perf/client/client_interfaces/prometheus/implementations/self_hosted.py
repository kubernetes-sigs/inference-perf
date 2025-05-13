from typing import List, Optional
import requests
from inference_perf.client.client_interfaces.prometheus.base import PrometheusMetricsCollector
from inference_perf.client.client_interfaces.prometheus.prometheus_metrics import PrometheusMetric
from inference_perf.config import SelfHostedPrometheusCollectorConfig


class SelfHostedPrometheusMetricsCollector(PrometheusMetricsCollector):
    config: SelfHostedPrometheusCollectorConfig

    def __init__(self, metrics: List[PrometheusMetric], config: SelfHostedPrometheusCollectorConfig):
        super().__init__(metrics=metrics)
        self.config = config

    async def query_metric(self, query: str, duration: float) -> Optional[float]:
        """
        Executes the given query on the Prometheus server and returns the result.

        Args:
        query: the PromQL query to execute

        Returns:
        The result of the query.
        """
        query_result = 0.0
        try:
            response = requests.get(f"{self.url}/api/v1/query", params={"query": query, "time": str(duration)})
            if response is None:
                print("Error executing query: %s" % (query))
                return query_result

            response.raise_for_status()
        except Exception as e:
            print("Error executing query: %s" % (e))
            return query_result

        # Check if the response is valid
        # Sample response:
        # {
        #     "status": "success",
        #     "data": {
        #         "resultType": "vector",
        #         "result": [
        #             {
        #                 "metric": {},
        #                 "value": [
        #                     1632741820.781,
        #                     "0.0000000000000000"
        #                 ]
        #             }
        #         ]
        #     }
        # }
        response_obj = response.json()
        if response_obj.get("status") != "success":
            print("Error executing query: %s" % (response_obj))
            return query_result

        data = response_obj.get("data", {})
        result = data.get("result", [])
        if len(result) > 0 and "value" in result[0]:
            if isinstance(result[0]["value"], list) and len(result[0]["value"]) > 1:
                # Return the value of the first result
                # The value is in the second element of the list
                # e.g. [1632741820.781, "0.0000000000000000"]
                # We need to convert it to float
                # and return it
                # Convert the value to float
                try:
                    query_result = float(result[0]["value"][1])
                except ValueError:
                    print("Error converting value to float: %s" % (result[0]["value"][1]))
                    return query_result
        return query_result
