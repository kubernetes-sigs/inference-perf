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
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from inference_perf.config import APIConfig, APIType, DataConfig, OTelTraceReplayConfig
from inference_perf.config import VisionArenaConfig
from inference_perf.datagen.dataset import cnn_dailymail_datagen, hf_sharegpt_datagen
from inference_perf.datagen.dataset import visionarena_datagen
from inference_perf.datagen.replay import otel_trace_replay_datagen
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.dataset import load_dataset_with_deadline

pytestmark = pytest.mark.timeout(10)


@pytest.fixture
def blocked_load() -> Iterator[Callable[..., Any]]:
    release = threading.Event()
    threads: list[threading.Thread] = []

    def load(*args: Any, **kwargs: Any) -> Any:
        thread = threading.current_thread()
        assert thread.daemon
        threads.append(thread)
        # Bound the fake hang too, so a broken deadline cannot strand the test runner.
        release.wait(5)
        return []

    yield load
    release.set()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()


def test_load_returns_dataset() -> None:
    rows = [{"text": "hello"}]
    assert load_dataset_with_deadline(lambda: rows, "test/data", 5) is rows


def test_loader_error_propagates() -> None:
    error = OSError("download failed")

    def load() -> None:
        raise error

    with pytest.raises(OSError) as exc:
        load_dataset_with_deadline(load, "test/data", 5)
    assert exc.value is error


def test_hung_load_times_out(blocked_load: Callable[..., Any]) -> None:
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="test/data.*0.05 seconds") as exc:
        load_dataset_with_deadline(blocked_load, "test/data", 0.05)
    assert time.monotonic() - start < 2
    assert "data.load_timeout" in str(exc.value)
    assert "HF_HUB_DISABLE_XET=1" in str(exc.value)


def test_null_disables_deadline() -> None:
    rows = [{"text": "hello"}]

    def load() -> list[dict[str, str]]:
        time.sleep(0.05)
        return rows

    assert load_dataset_with_deadline(load, "test/data", None) is rows


def test_stream_is_not_consumed() -> None:
    consumed = False

    def rows() -> Iterator[str]:
        nonlocal consumed
        consumed = True
        yield "hello"

    stream = rows()
    assert load_dataset_with_deadline(lambda: stream, "test/data", 5) is stream
    assert not consumed
    assert next(stream) == "hello"


def test_default_timeout() -> None:
    assert DataConfig().load_timeout == 300.0


@pytest.mark.parametrize("value", [0.001, 300.0, None])
def test_accepts_positive_or_null_timeout(value: float | None) -> None:
    assert DataConfig(load_timeout=value).load_timeout == value


@pytest.mark.parametrize("value", [0, -1, math.inf, -math.inf, math.nan])
def test_rejects_invalid_timeout(value: float) -> None:
    with pytest.raises(ValidationError, match="load_timeout"):
        DataConfig(load_timeout=value)


@pytest.mark.parametrize("stall", ["load", "first_row"])
@pytest.mark.parametrize("source", ["sharegpt", "cnn", "visionarena", "otel"])
def test_generators_apply_configured_deadline(
    source: str, stall: str, monkeypatch: pytest.MonkeyPatch, blocked_load: Callable[..., Any]
) -> None:
    if source == "otel" and stall == "first_row":
        pytest.skip("OTel loads eagerly")

    def rows() -> Iterator[Any]:
        blocked_load()
        yield {}

    def load(*args: Any, **kwargs: Any) -> Any:
        return rows() if stall == "first_row" else blocked_load()

    config = DataConfig(load_timeout=0.05)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="data.load_timeout"):
        if source == "sharegpt":
            monkeypatch.setattr(hf_sharegpt_datagen, "load_dataset", load)
            hf_sharegpt_datagen.HFShareGPTDataGenerator(APIConfig(), config, None)
        elif source == "cnn":
            monkeypatch.setattr(cnn_dailymail_datagen, "load_dataset", load)
            # CNN requires a tokenizer, but loading times out before it is used.
            tokenizer = CustomTokenizer.__new__(CustomTokenizer)
            cnn_dailymail_datagen.CNNDailyMailDataGenerator(APIConfig(), config, tokenizer)
        elif source == "visionarena":
            monkeypatch.setattr(visionarena_datagen, "load_dataset", load)
            config.visionarena = VisionArenaConfig()
            visionarena_datagen.VisionArenaDataGenerator(APIConfig(type=APIType.Chat), config, None)
        else:
            monkeypatch.setattr(otel_trace_replay_datagen, "load_dataset", load)
            config.otel_trace_replay = OTelTraceReplayConfig(hf_dataset_path="test/traces")
            otel_trace_replay_datagen.OTelTraceReplayDataGenerator(APIConfig(type=APIType.Chat), config, None)
    assert time.monotonic() - start < 2


def test_otel_passes_dataset_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from datasets import Dataset

    rows = Dataset.from_list([{"trace_id": "test", "spans": []}])
    calls: list[tuple[str, dict[str, Any]]] = []

    def load(path: str, **kwargs: Any) -> Dataset:
        calls.append((path, kwargs))
        return rows

    monkeypatch.setattr(otel_trace_replay_datagen, "load_dataset", load)
    options = {"path": "test/traces", "split": "test", "revision": "v1"}
    assert otel_trace_replay_datagen._download_hf_dataset(options, None) is rows
    assert calls == [("test/traces", {"split": "test", "revision": "v1"})]
    assert options == {"path": "test/traces", "split": "test", "revision": "v1"}


@pytest.mark.parametrize("source", ["sharegpt", "cnn"])
@pytest.mark.parametrize("timeout", [5.0, None])
def test_streaming_startup_primes_once_and_reloads(
    source: str, timeout: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumed: list[int] = []

    def load(*args: Any, **kwargs: Any) -> Iterator[int]:
        for value in range(3):
            consumed.append(value)
            yield value

    generator: hf_sharegpt_datagen.HFShareGPTDataGenerator | cnn_dailymail_datagen.CNNDailyMailDataGenerator
    config = DataConfig(load_timeout=timeout)
    if source == "sharegpt":
        monkeypatch.setattr(hf_sharegpt_datagen, "load_dataset", load)
        generator = hf_sharegpt_datagen.HFShareGPTDataGenerator(APIConfig(), config, None)
        attribute = "sharegpt_dataset"
    else:
        monkeypatch.setattr(cnn_dailymail_datagen, "load_dataset", load)
        tokenizer = CustomTokenizer.__new__(CustomTokenizer)
        generator = cnn_dailymail_datagen.CNNDailyMailDataGenerator(APIConfig(), config, tokenizer)
        attribute = "cnn_dailymail_dataset"
    assert consumed == [0]
    assert next(getattr(generator, attribute)) == 1
    restored = type(generator).__new__(type(generator))
    restored.__setstate__(generator.__getstate__())
    assert consumed == [0, 1]
    restored._ensure_dataset_loaded()
    assert consumed == [0, 1, 0]
    assert next(getattr(restored, attribute)) == 1


@pytest.mark.parametrize("timeout", [5.0, None])
def test_visionarena_stops_reading_when_pool_is_full(timeout: float | None, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    def load(*args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield {"conversation": [{"role": "user", "content": "describe"}], "images": [Image.new("RGB", (1, 1))]}
        raise AssertionError("Read beyond the configured pool size")

    monkeypatch.setattr(visionarena_datagen, "load_dataset", load)
    config = DataConfig(load_timeout=timeout, visionarena=VisionArenaConfig(num_rows=1))
    generator = visionarena_datagen.VisionArenaDataGenerator(APIConfig(type=APIType.Chat), config, None)
    assert len(generator._pool) == 1


def test_visionarena_deadline_covers_later_pool_rows(
    monkeypatch: pytest.MonkeyPatch, blocked_load: Callable[..., Any]
) -> None:
    from PIL import Image

    def load(*args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield {
            "conversation": [{"role": "user", "content": "describe"}],
            "images": [Image.new("RGB", (1, 1))],
        }
        blocked_load()

    monkeypatch.setattr(visionarena_datagen, "load_dataset", load)
    config = DataConfig(load_timeout=0.05, visionarena=VisionArenaConfig(num_rows=2))
    with pytest.raises(TimeoutError, match="data.load_timeout"):
        visionarena_datagen.VisionArenaDataGenerator(APIConfig(type=APIType.Chat), config, None)
