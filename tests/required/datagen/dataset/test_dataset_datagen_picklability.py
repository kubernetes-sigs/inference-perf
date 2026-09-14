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
"""Regression tests for #589.

HFShareGPTDataGenerator, CNNDailyMailDataGenerator, BillsumConversationsDataGenerator
and InfinityInstructDataGenerator each store a live iterator over a streaming
HuggingFace dataset on the instance. Worker processes started under spawn or
forkserver (Python 3.14's default) pickle the whole Worker, including its
datagen, so the unpicklable iterator crashed the worker with no useful
traceback. Wrapping it in itertools.cycle didn't help: that pickling support
is itself going away in 3.14.

The restored copy also must not reload the dataset eagerly: a Worker process
only reaches its own copy of one of these generators through
LazyLoadDataMixin.get_request(), which is a no-op for all four (the parent
process is the one that calls get_data() and puts materialized requests on
the queue), so an eager reload on every worker spawn would be wasted I/O,
network round trips included for the default Hub datasets.
"""

import json
import pathlib
import pickle

import pytest

from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.datagen.dataset.cnn_dailymail_datagen import CNNDailyMailDataGenerator
from inference_perf.datagen.dataset.hf_billsum_datagen import BillsumConversationsDataGenerator
from inference_perf.datagen.dataset.hf_sharegpt_datagen import HFShareGPTDataGenerator
from inference_perf.datagen.dataset.infinity_instruct_datagen import InfinityInstructDataGenerator


@pytest.fixture(autouse=True)
def _offline_hf_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # All datasets here load from a local file, but datasets still reaches out
    # to check the packaged "json" loader script unless told not to. datasets
    # and huggingface_hub read HF_HUB_OFFLINE/HF_DATASETS_OFFLINE into module
    # constants once at import time, so setting the environment variable here
    # is too late: patch the resolved constants directly.
    monkeypatch.setattr("datasets.config.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("datasets.config.HF_DATASETS_OFFLINE", True)


def _write_conversation_rows(path: pathlib.Path, *rows: str) -> None:
    # Each row becomes one dataset record, with the given text as the human
    # turn. A row is fully consumed by one next() call, which matters for
    # generators that don't wrap the dataset in itertools.cycle: priming the
    # restored iterator takes a next() call of its own, so a test calling
    # next() again needs a second row to still be there.
    path.write_text(
        json.dumps([{"conversations": [{"from": "human", "value": row}, {"from": "gpt", "value": "ack"}]} for row in rows])
    )


def test_hf_sharegpt_datagen_survives_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    data_file = tmp_path / "sharegpt.json"
    _write_conversation_rows(data_file, "hi")

    generator = HFShareGPTDataGenerator(APIConfig(type=APIType.Completion), DataConfig(path=str(data_file)), None)

    restored: HFShareGPTDataGenerator = pickle.loads(pickle.dumps(generator))
    assert restored._dataset_ready is False

    restored._ensure_dataset_loaded()
    assert restored._dataset_ready is True
    assert next(restored.sharegpt_dataset)["conversations"][0]["value"] == "hi"


def test_cnn_dailymail_datagen_survives_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    data_file = tmp_path / "cnn_dailymail.json"
    data_file.write_text(json.dumps({"article": "an article", "highlights": "a summary"}))

    generator = CNNDailyMailDataGenerator(
        APIConfig(type=APIType.Completion),
        DataConfig(path=str(data_file)),
        object(),  # type: ignore[arg-type]
    )

    restored: CNNDailyMailDataGenerator = pickle.loads(pickle.dumps(generator))
    assert restored._dataset_ready is False

    restored._ensure_dataset_loaded()
    assert restored._dataset_ready is True
    assert next(restored.cnn_dailymail_dataset)["article"] == "an article"


def test_billsum_datagen_survives_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    data_file = tmp_path / "billsum.json"
    _write_conversation_rows(data_file, "summarize this bill", "summarize that other bill")

    generator = BillsumConversationsDataGenerator(APIConfig(type=APIType.Completion), DataConfig(path=str(data_file)), None)

    restored: BillsumConversationsDataGenerator = pickle.loads(pickle.dumps(generator))
    assert restored._dataset_ready is False

    restored._ensure_dataset_loaded()
    assert restored._dataset_ready is True
    # _ensure_dataset_loaded() primes the restored iterator with its own
    # next() call, so this next() lands on the second row.
    assert next(restored.billsum_dataset)["conversations"][0]["value"] == "summarize that other bill"


def test_infinity_instruct_datagen_survives_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    data_file = tmp_path / "infinity_instruct.json"
    _write_conversation_rows(data_file, "hi", "hi again")

    generator = InfinityInstructDataGenerator(APIConfig(type=APIType.Completion), DataConfig(path=str(data_file)), None)

    restored: InfinityInstructDataGenerator = pickle.loads(pickle.dumps(generator))
    assert restored._dataset_ready is False

    restored._ensure_dataset_loaded()
    assert restored._dataset_ready is True
    # _ensure_dataset_loaded() primes the restored iterator with its own
    # next() call, so this next() lands on the second row.
    assert next(restored.infinity_instruct_dataset)["conversations"][0]["value"] == "hi again"
