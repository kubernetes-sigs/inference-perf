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
import logging
from pydantic import HttpUrl
import requests
from inference_perf.client.metricsclient.prometheus_client.base import PrometheusMetricsClient
from inference_perf.config import PrometheusClientConfig
import google.auth

logger = logging.getLogger(__name__)


class GoogleManagedPrometheusMetricsClient(PrometheusMetricsClient):
    def __init__(self, config: PrometheusClientConfig) -> None:
        if config.google_managed is None:
            raise Exception("google managed prometheus config missing")
        self.google_managed_prometheus_config = config.google_managed
        # Creates a credentials object from the default service account file
        # Assumes that script has appropriate default credentials set up, ref:
        # https://googleapis.dev/python/google-auth/latest/user-guide.html#application-default-credentials
        credentials, project_id = google.auth.default()  # type: ignore[no-untyped-call]
        self.credentials = credentials
        self.project_id = project_id
        config.url = HttpUrl(f"https://monitoring.googleapis.com/v1/projects/{self.project_id}/location/global/prometheus")
        super().__init__(config)

    def execute_query(self, query: str, eval_time: str) -> float:
        """
        Executes the given query on the Prometheus server and returns the result.

        Args:
        query: the PromQL query to execute
        eval_time: the time at which the query is evaluated, used to ensure we are querying the correct time range

        Returns:
        The result of the query.
        """
        query_result = 0.0

        # Prepare an authentication request - helps format the request auth token
        auth_req = google.auth.transport.requests.Request()

        self.credentials.refresh(auth_req)
        headers = {"Authorization": "Bearer " + self.credentials.token}

        try:
            logger.info(f"Making PromQL query: '{query}'")
            response = requests.get(
                url=f"{self.url}/api/v1/query", headers=headers, params={"query": query, "time": eval_time}
            )
            if response is None:
                logger.error("Error executing query: %s" % (query))
                return query_result

            response.raise_for_status()
        except Exception as e:
            logger.error("Error executing query: %s" % (e))
            return query_result

        response_obj = response.json()
        if response_obj.get("status") != "success":
            logger.error("Error executing query: %s" % (response_obj))
            return query_result

        data = response_obj.get("data", {})
        result = data.get("result", [])
        if len(result) > 0 and "value" in result[0]:
            if isinstance(result[0]["value"], list) and len(result[0]["value"]) > 1:
                try:
                    query_result = round(float(result[0]["value"][1]), 6)
                except ValueError:
                    logger.error("Error converting value to float: %s" % (result[0]["value"][1]))
                    return query_result
        return query_result
