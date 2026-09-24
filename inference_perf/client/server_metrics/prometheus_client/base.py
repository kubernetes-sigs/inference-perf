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
import logging
import re
import time
from typing import Any, Optional
import requests
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.config import PrometheusClientConfig
from ..base import ServerMetricsClient, PerfRuntimeParameters, ModelServerMetrics

PROMETHEUS_SCRAPE_BUFFER_SEC = 2
# Match only PromQL metric selectors, leaving colons in label values untouched.
_PROMQL_METRIC_SELECTOR = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*(?=\{)")

logger = logging.getLogger(__name__)


class PrometheusMetricsClient(ServerMetricsClient):
    def __init__(self, config: PrometheusClientConfig) -> None:
        if config:
            if not config.url:
                raise Exception("prometheus url missing")
            self.query_url = config.url.unicode_string().rstrip("/") + "/api/v1/query"
            logger.debug(f"Prometheus metrics client configured, querying metrics from '{self.query_url}'")
            self.scrape_interval = config.scrape_interval or 30
            self.bearer_token = config.bearer_token.get_secret_value() if config.bearer_token else None
            self.verify_ssl = config.verify_ssl
            self.extra_headers = dict(config.headers) if config.headers else {}
        else:
            raise Exception("prometheus config missing")

    def wait(self) -> None:
        """
        Waits for the Prometheus server to scrape the metrics.
        We have added a buffer of 5 seconds to the scrape interval to ensure that metrics for even the last request are collected.
        """
        wait_time = self.scrape_interval + PROMETHEUS_SCRAPE_BUFFER_SEC
        time.sleep(wait_time)

    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> Optional[ModelServerMetrics]:
        """
        Collects the summary metrics for the given Perf Benchmark run.

        Args:
        runtime_parameters: The runtime parameters containing details about the Perf Benchmark like the duration and model server client

        Returns:
        A ModelServerMetrics object containing the summary metrics.
        """
        if runtime_parameters is None:
            logger.warning("Perf Runtime parameters are not set, skipping metrics collection")
            return None

        # Get the duration and model server client from the runtime parameters
        query_eval_time = time.time()
        query_duration = query_eval_time - runtime_parameters.start_time

        return self.get_model_server_metrics(runtime_parameters.model_server_metrics, query_duration, query_eval_time)

    def collect_metrics_for_stage(
        self, runtime_parameters: PerfRuntimeParameters, stage_id: int
    ) -> Optional[ModelServerMetrics]:
        """
        Collects the summary metrics for a specific stage.

        Args:
        runtime_parameters: The runtime parameters containing details about the Perf Benchmark like the duration and model server client
        stage_id: The ID of the stage for which to collect metrics

        Returns:
        A ModelServerMetrics object containing the summary metrics for the specified stage.
        """
        if runtime_parameters is None:
            logger.warning("Perf Runtime parameters are not set, skipping metrics collection")
            return None

        if runtime_parameters.stages is None or stage_id not in runtime_parameters.stages:
            logger.warning(
                f"Stage ID {stage_id} is not present in the runtime parameters, skipping metrics collection for this stage"
            )
            return None

        # Get the query evaluation time and duration for the stage
        # The query evaluation time is the end time of the stage plus the scrape interval and a buffer to ensure metrics are collected
        # Duration is calculated as the difference between the eval time and start time of the stage
        logger.debug(f"runtime parameters for stage {stage_id}: {runtime_parameters}")
        query_eval_time = runtime_parameters.stages[stage_id].end_time + self.scrape_interval + PROMETHEUS_SCRAPE_BUFFER_SEC
        query_duration = query_eval_time - runtime_parameters.stages[stage_id].start_time
        return self.get_model_server_metrics(runtime_parameters.model_server_metrics, query_duration, query_eval_time)

    def get_model_server_metrics(
        self,
        metrics_metadata: BaseMetrics,
        query_duration: float,
        query_eval_time: float,
    ) -> Optional[ModelServerMetrics]:
        """
        Collects the summary metrics for the given Model Server Client and query duration.

        Args:
        metrics_metadata: The model server metrics descriptors to query
        query_duration: The duration for which to collect metrics
        query_eval_time: The time at which the query is evaluated, used to ensure we are querying the correct time range

        Returns:
        A ModelServerMetrics object containing the summary metrics.
        """

        def execute(query: str) -> float:
            eval_time = str(query_eval_time)
            result = self.execute_query(query, eval_time)
            if result is None:
                fallback_query = _PROMQL_METRIC_SELECTOR.sub(lambda match: match.group().replace(":", "_"), query)
                if fallback_query != query:
                    result = self.execute_query(fallback_query, eval_time)
            return 0.0 if result is None else result

        # Iterating the metadata yields (target_field, metric) pairs; each metric owns its
        # query+parse (collect), with the container's shared label filters applied. Building the
        # dict and validating it through Pydantic enforces the field's declared result type.
        filters = metrics_metadata.filters
        pairs = list(metrics_metadata)
        # Validation ignores extra keys, so a declaration targeting a nonexistent field would
        # run its queries and silently drop the results; fail before querying instead.
        unknown = sorted(field for field, _ in pairs if field not in ModelServerMetrics.model_fields)
        if unknown:
            raise ValueError(f"Metrics declared for unknown ModelServerMetrics field(s): {', '.join(unknown)}")
        collected = {field: metric.collect(execute, query_duration, filters) for field, metric in pairs}
        return ModelServerMetrics.model_validate(collected)

    def execute_query(self, query: str, eval_time: str) -> Optional[float]:
        """
        Executes the given query on the Prometheus server and returns the result.

        Args:
        query: the PromQL query to execute
        eval_time: the time at which the query is evaluated, used to ensure we are querying the correct time range

        Returns:
        The first query result, or None when a successful query returns no series.
        """
        query_result = 0.0
        try:
            logger.debug(f"making PromQL query: '{query}'")
            response = requests.get(
                self.query_url, headers=self.get_headers(), params={"query": query, "time": eval_time}, verify=self.verify_ssl
            )
            if response is None:
                logger.error("error executing query: %s" % (query))
                return query_result

            response.raise_for_status()
        except Exception as e:
            logger.error("error executing query: %s" % (e))
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
        logger.debug(f"got result for query '{query}': {response_obj}")
        if response_obj.get("status") != "success":
            logger.error("error executing query: %s" % (response_obj))
            return query_result

        data = response_obj.get("data", {})
        result = data.get("result", [])
        if not result:
            logger.debug(f"query '{query}' returned no series")
            return None
        if len(result) > 0 and "value" in result[0]:
            if isinstance(result[0]["value"], list) and len(result[0]["value"]) > 1:
                # Return the value of the first result
                # The value is in the second element of the list
                # e.g. [1632741820.781, "0.0000000000000000"]
                # We need to convert it to float
                # and return it
                # Convert the value to float
                try:
                    query_result = round(float(result[0]["value"][1]), 6)
                except ValueError:
                    logger.error("error converting value to float: %s" % (result[0]["value"][1]))
                    return query_result
        logger.debug(f"inferred result from query '{query}': {query_result}")
        return query_result

    def get_headers(self) -> dict[str, Any]:
        headers: dict[str, Any] = dict(self.extra_headers)
        if self.bearer_token and "authorization" not in {str(k).lower() for k in headers}:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers
