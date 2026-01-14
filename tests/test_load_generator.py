import pytest
from inference_perf.config import LoadConfig, TrafficSplitConfig, StandardLoadStage
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import APIConfig, DataConfig
from collections import Counter
from inference_perf.apis import InferenceAPIData


class MockCounterServerClient:
    def __init__(self):
        self.models: Counter[str] = Counter()

    async def process_request(self, data: InferenceAPIData, stage_id: int, scheduled_time: float, model_name: str | None):
        self.models[model_name] += 1


@pytest.mark.asyncio
async def test_traffic_split_distribution():
    """Test that LoadGenerator distributes requests according to traffic split weights."""

    traffic_split = [
        TrafficSplitConfig(model_name="adapter1", weight=0.7),
        TrafficSplitConfig(model_name="adapter2", weight=0.3),
    ]

    rate = 500
    duration = 2
    total_requests = rate * duration - 1

    load_config = LoadConfig(
        stages=[StandardLoadStage(rate=rate, duration=duration)],
        traffic_split=traffic_split,
        num_workers=0,
    )

    datagen = MockDataGenerator(APIConfig(), DataConfig(), None)
    load_gen = LoadGenerator(datagen, load_config)

    mock_client = MockCounterServerClient()

    await load_gen.run(client=mock_client)

    assert sum(mock_client.models.values()) == total_requests
    assert abs((mock_client.models["adapter1"] / total_requests) - 0.7) < 0.05
    assert abs((mock_client.models["adapter2"] / total_requests) - 0.3) < 0.05
