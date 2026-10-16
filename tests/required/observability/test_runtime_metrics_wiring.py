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
"""Wiring of the runtime metrics hub into a real run.

The hub itself is unit-tested in test_registry.py. These tests cover the
subscription points a benchmark run goes through: request collector observers
(local and multiprocess, notified once per metric in the aggregating process),
stage transitions and in-flight sampling from the load generator, and the
config surface that turns the HTTP endpoint on.
"""

import multiprocessing as mp
import pickle
import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from prometheus_client import Counter, Gauge  # noqa: TID251 (throwaway metrics for these tests)

from inference_perf.apis import InferenceAPIData
from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.client.modelserver.mock_client import MockModelServerClient
from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    DataConfig,
    DataGenType,
    Distribution,
    LoadConfig,
    LoadType,
    ObservabilityConfig,
    RuntimeMetricsConfig,
    StandardLoadStage,
    read_config,
)
from inference_perf.datagen import DataGenerator
from inference_perf.datagen.synthetic.random_datagen import RandomDataGenerator
from inference_perf.main import log_metrics_endpoint_failure
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector import (
    LocalRequestMetricCollector,
    MultiprocessRequestMetricCollector,
)
from inference_perf.observability.metrics import MetricSpec, RunContext, StageContext, build_metrics
from inference_perf.observability.metrics.prometheus import DEFAULT_PORT
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.utils.cli_parser import add_global_args, add_pydantic_args, unflatten_dict
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# Match the start method used in production (main.py) so objects are inherited
# rather than pickled into worker processes.
if sys.platform == "darwin":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass


def _metric(stage_id: int = 0) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="r",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1))),
        error=None,
    )


class _RecordingObserver:
    def __init__(self) -> None:
        self.events: List[Any] = []
        self.contexts: List[StageContext] = []

    def __call__(self, metric: RequestLifecycleMetric) -> None:
        self.events.append(metric)

    def on_stage_start(self, context: StageContext) -> None:
        self.events.append(("start", context.stage_id))
        self.contexts.append(context)

    def on_stage_end(self, context: StageContext) -> None:
        self.events.append(("end", context.stage_id))
        self.contexts.append(context)


class TestCollectorObservers(unittest.IsolatedAsyncioTestCase):
    def test_local_collector_notifies_each_observer_per_metric(self) -> None:
        collector = LocalRequestMetricCollector()
        first, second = _RecordingObserver(), _RecordingObserver()
        collector.add_observer(first)
        collector.add_observer(second)

        collector.record_metric(_metric())
        collector.record_metric(_metric())

        self.assertEqual(len(collector.get_metrics()), 2)
        self.assertEqual(len(first.events), 2)
        self.assertEqual(len(second.events), 2)

    async def test_multiprocess_collector_notifies_in_the_drain_loop(self) -> None:
        collector = MultiprocessRequestMetricCollector()
        observer = _RecordingObserver()
        collector.add_observer(observer)

        async with collector.start():
            # record_metric only enqueues (this is what workers call); nothing
            # is observed until the parent drain loop ingests the item.
            collector.record_metric(_metric(stage_id=3))
            collector.record_metric(_metric(stage_id=3))

        self.assertEqual(len(collector.get_metrics()), 2)
        self.assertEqual([m.stage_id for m in observer.events], [3, 3])

    def test_pickling_a_collector_drops_its_observers(self) -> None:
        # Observers hold locks/sockets and only run in the aggregating process;
        # a copy shipped to a forkserver/spawn worker must not carry them.
        collector = LocalRequestMetricCollector()
        collector.add_observer(_RecordingObserver())
        clone = pickle.loads(pickle.dumps(collector))
        self.assertEqual(clone._observers, [])
        self.assertEqual(len(collector._observers), 1)


class TestHubLifecycleHooks(unittest.TestCase):
    def test_run_and_stage_hooks_receive_context_and_stage_ids(self) -> None:
        seen: List[Any] = []

        def _run_start(gauge: Gauge, context: RunContext) -> None:
            seen.append(("run", len(context.config.load.stages)))
            gauge.set_function(context.in_flight_requests)

        def _stage_start(counter: Counter, context: StageContext) -> None:
            seen.append(("start", context.stage_id))

        def _stage_end(counter: Counter, context: StageContext) -> None:
            seen.append(("end", context.stage_id))

        specs = (
            MetricSpec(name="test_in_flight", documentation="live probe", metric_type=Gauge, on_run_start=_run_start),
            MetricSpec(
                name="test_stage_events",
                documentation="stage hooks",
                metric_type=Counter,
                on_stage_start=_stage_start,
                on_stage_end=_stage_end,
            ),
        )
        config = Config(load=LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)] * 3))
        hub = build_metrics(config, specs=specs)

        in_flight = 7
        hub.on_run_start(RunContext(config=config, in_flight_requests=lambda: in_flight))
        hub.on_stage_start(StageContext(stage_id=0))
        hub.on_stage_end(StageContext(stage_id=0))

        self.assertEqual(seen, [("run", 3), ("start", 0), ("end", 0)])
        self.assertEqual(hub.registry.get_sample_value("test_in_flight"), 7.0)
        in_flight = 2
        self.assertEqual(hub.registry.get_sample_value("test_in_flight"), 2.0)

    def test_on_run_start_without_context_uses_build_config(self) -> None:
        seen: List[int] = []

        def _run_start(gauge: Gauge, context: RunContext) -> None:
            seen.append(len(context.config.load.stages))
            seen.append(context.in_flight_requests())

        specs = (MetricSpec(name="test_ctx", documentation="ctx", metric_type=Gauge, on_run_start=_run_start),)
        hub = build_metrics(Config(load=LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)] * 2)), specs=specs)
        hub.on_run_start()
        self.assertEqual(seen, [2, 0])


# Parses argv the way main_cli does, global args first and then every Config field
# as a flag, and returns the Config the run would use.
def _config_from_argv(argv: List[str]) -> Config:
    parser = ArgumentParser()
    base_args = add_global_args(parser)
    add_pydantic_args(parser, Config)
    args = parser.parse_args(argv)
    overrides = unflatten_dict({k: v for k, v in vars(args).items() if k not in base_args})
    return read_config(args.config_file, overrides)


class TestObservabilityConfig(unittest.TestCase):
    # A Config with no observability block.
    # Expects metrics to be None, so a run that never asked for the endpoint opens no port.
    def test_endpoint_is_off_by_default(self) -> None:
        self.assertIsNone(Config().observability.metrics)

    # observability.metrics: {} in YAML, the shortest way to turn the endpoint on.
    # Expects it on at 9464, the server's default port.
    def test_empty_block_turns_the_endpoint_on_at_the_server_port(self) -> None:
        metrics = Config.model_validate({"observability": {"metrics": {}}}).observability.metrics
        assert metrics is not None
        # The config default is a literal to avoid a config <-> observability
        # import cycle. This pins it to the server's default.
        self.assertEqual(metrics.port, DEFAULT_PORT)

    # YAML asking for an ephemeral port.
    # Expects it to parse as port 0.
    def test_yaml_shape_round_trips(self) -> None:
        config = Config.model_validate({"observability": {"metrics": {"port": 0}}})
        self.assertEqual(config.observability, ObservabilityConfig(metrics=RuntimeMetricsConfig(port=0)))

    # --observability.metrics.port 9465 with no config file, the way a docker run turns metrics on.
    # Expects the flag alone to create the block, so the endpoint runs on 9465.
    def test_port_flag_alone_turns_the_endpoint_on(self) -> None:
        metrics = _config_from_argv(["--observability.metrics.port", "9465"]).observability.metrics
        assert metrics is not None
        self.assertEqual(metrics.port, 9465)

    # The root config.yml (README quickstart, Docker image), the Helm chart's values, and the
    # deploy/ Job, which runs that same config.yml with its own args.
    # Expects no endpoint for the quickstart and 9464 for both in-cluster paths.
    def test_only_the_in_cluster_configs_serve_metrics(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        self.assertIsNone(read_config(str(repo / "config.yml")).observability.metrics)
        chart = yaml.safe_load((repo / "deploy" / "inference-perf" / "values.yaml").read_text())
        job = yaml.safe_load((repo / "deploy" / "manifests.yaml").read_text())
        # The Job mounts config.yml from a ConfigMap. Read the repo copy it is made from.
        job_args = [
            str(repo / "config.yml") if arg == "/etc/config/config.yml" else arg
            for arg in job["spec"]["template"]["spec"]["containers"][0]["args"]
        ]
        for config in (Config.model_validate(chart["config"]), _config_from_argv(job_args)):
            metrics = config.observability.metrics
            assert metrics is not None
            self.assertEqual(metrics.port, 9464)


class TestMetricsEndpointFailureLogging(unittest.TestCase):
    # The endpoint fails to bind on 9464 with "Address already in use".
    # Expects one ERROR naming the port, since the endpoint only runs when someone asked for it.
    def test_port_collision_is_an_error(self) -> None:
        with self.assertLogs("inference_perf.main", level="WARNING") as cm:
            log_metrics_endpoint_failure(RuntimeMetricsConfig(), OSError("Address already in use"))
        self.assertEqual([r.levelname for r in cm.records], ["ERROR"])
        self.assertIn("port 9464", cm.output[0])


class TestLoadGeneratorStageObserver(unittest.IsolatedAsyncioTestCase):
    """The load generator reports stage transitions and in-flight requests to its observer."""

    def _in_process_loadgen(self, observer: _RecordingObserver) -> LoadGenerator:
        datagen = MagicMock(spec=DataGenerator)
        datagen.trace = None
        data = MagicMock(spec=InferenceAPIData)
        data.preferred_worker_id = -1
        datagen.get_data.return_value = iter([data, data])
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=0,
            stages=[StandardLoadStage(rate=2, duration=1)],
            circuit_breakers=[],
        )
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            return LoadGenerator(datagen, load_config, stage_observer=observer)

    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    async def test_in_process_run_reports_stage_transitions_and_in_flight(self, _sleep: AsyncMock) -> None:
        observer = _RecordingObserver()
        loadgen = self._in_process_loadgen(observer)
        sampled: List[int] = []

        async def _process_request(*args: Any, **kwargs: Any) -> None:
            sampled.append(loadgen.in_flight_requests())

        client = AsyncMock(spec=ModelServerClient)
        client.process_request = AsyncMock(side_effect=_process_request)
        timer = MagicMock()
        timer.start_timer.return_value = iter([0.1, 0.2])

        self.assertEqual(loadgen.in_flight_requests(), 0)
        with (
            patch("inference_perf.loadgen.load_generator.LazyLoadDataMixin.get_request", side_effect=lambda dg, d: d),
            patch("inference_perf.loadgen.load_generator.time.perf_counter", return_value=0.0),
            patch.object(loadgen, "get_timer", return_value=timer),
        ):
            await loadgen.run(client)

        self.assertEqual(observer.events, [("start", 0), ("end", 0)])
        # The bar's denominator has to travel with the transition, and the
        # finished probe has to read the requests that actually completed.
        self.assertEqual([c.planned_requests for c in observer.contexts], [2, 2])
        self.assertEqual(observer.contexts[-1].requests_finished(), 2)
        self.assertEqual(sampled, [1, 1], "each request should see itself in flight while it runs")
        self.assertEqual(loadgen.in_flight_requests(), 0)

    async def test_multiprocess_run_feeds_stage_observer_and_collector_observers(self) -> None:
        num_requests, num_workers = 6, 2
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        data_config = DataConfig(
            type=DataGenType.Random,
            input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=num_requests),
            output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=num_requests),
        )
        datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer())

        collector = MultiprocessRequestMetricCollector()
        hub = build_metrics(Config())
        collector.add_observer(hub.observe_request)
        client = MockModelServerClient(collector, api_config, mock_latency=0)

        observer = _RecordingObserver()
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=num_workers,
            worker_max_concurrency=10,
            stages=[StandardLoadStage(rate=num_requests, duration=1), StandardLoadStage(rate=num_requests, duration=1)],
            base_seed=42,
        )
        loadgen = LoadGenerator(datagen, load_config, stage_observer=observer)
        hub.on_run_start(RunContext(config=Config(load=load_config), in_flight_requests=loadgen.in_flight_requests))

        async with collector.start():
            await loadgen.mp_run(client)

        self.assertEqual(len(collector.get_metrics()), 2 * num_requests)
        self.assertEqual(observer.events, [("start", 0), ("end", 0), ("start", 1), ("end", 1)])
        self.assertEqual([c.planned_requests for c in observer.contexts], [num_requests] * 4)
        self.assertEqual(observer.contexts[-1].requests_finished(), num_requests)
        self.assertEqual(observer.contexts[-1].requests_skipped(), 0)
        counter = "inference_perf_requests_total"
        self.assertEqual(hub.registry.get_sample_value(counter, {"stage": "0", "status": "success"}), float(num_requests))
        self.assertEqual(hub.registry.get_sample_value(counter, {"stage": "1", "status": "success"}), float(num_requests))
        self.assertEqual(loadgen.in_flight_requests(), 0)


class _DummyHFTokenizer:
    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def decode(self, tokens: List[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)

    def encode(self, text: str) -> List[int]:
        try:
            return [int(t) for t in text.split()]
        except ValueError:
            return list(range(10, 10010))


class _DummyCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return _DummyHFTokenizer()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split()) if text else 0


if __name__ == "__main__":
    unittest.main()
