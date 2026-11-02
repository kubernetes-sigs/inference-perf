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

import io
import itertools
import json
import os
import pickle
import threading
import tracemalloc
from collections.abc import Buffer, Generator
from pathlib import Path
from unittest.mock import Mock

import ijson
import pytest

from inference_perf.apis import AnthropicMessagesAPIData, ChatCompletionAPIData, CompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution
from inference_perf.datagen.dataset import hf_sharegpt_datagen as sharegpt


# Build one human prompt and one assistant reply for the parser fixtures.
def conversation(prompt: str) -> dict[str, list[dict[str, str]]]:
    return {"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": "reply"}]}


@pytest.mark.parametrize("json_lines", [False, True])
@pytest.mark.parametrize("source", ["file", "directory", "hub"])
# Three rows from a file, directory, or Hub repeat in input order after
# the existing first-row prime; Hub downloads once and completed files close.
def test_two_passes_preserve_order_and_priming(
    source: str, json_lines: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [conversation(text) for text in ("first", "second", "third")]

    # Serialize rows as an array or JSON Lines, with leading whitespace.
    def encode(data: list[dict[str, list[dict[str, str]]]]) -> str:
        return " \n\t" + ("\n".join(map(json.dumps, data)) if json_lines else json.dumps(data))

    config = DataConfig()
    streams: list[io.BytesIO] = []
    download = Mock(return_value=str(tmp_path / "cached.json"))
    open_file = Mock()
    if source == "hub":
        # Each cached-file open returns a fresh stream of the same three rows.
        def open_stream(*args: object, **kwargs: object) -> io.BytesIO:
            stream = io.BytesIO(encode(rows).encode())
            streams.append(stream)
            return stream

        open_file.side_effect = open_stream
        monkeypatch.setattr(sharegpt, "hf_hub_download", download)
        monkeypatch.setattr(sharegpt, "open", open_file, raising=False)
    elif source == "file":
        path = tmp_path / "rows.json"
        path.write_text(encode(rows))
        config.path = str(path)
    else:
        (tmp_path / "b.json").write_text(encode(rows[:1]))
        (tmp_path / "a.json").write_text(encode(rows[1:]))
        (tmp_path / "ignored.txt").write_text("not json")
        # Retain the existing os.listdir order, even when it is not alphabetical.
        monkeypatch.setattr(os, "listdir", lambda path: ["b.json", "ignored.txt", "a.json"])
        config.path = str(tmp_path)

    generator = sharegpt.HFShareGPTDataGenerator(APIConfig(), config, None)
    assert list(itertools.islice(generator.sharegpt_dataset, 6)) == rows[1:] + rows + rows[:1]
    if source == "hub":
        download.assert_called_once_with(
            repo_id=sharegpt.SHAREGPT_HF_DATASET_URL,
            filename=sharegpt.SHAREGPT_HF_DATAFILES_PATH,
            repo_type="dataset",
        )
        assert open_file.call_count == 3
        open_file.assert_called_with(str(tmp_path / "cached.json"), "rb", buffering=0)
        assert all(stream.closed for stream in streams[:-1])
        assert isinstance(generator.sharegpt_dataset, Generator)
        generator.sharegpt_dataset.close()
        assert streams[-1].closed


class BoundedReadStream(io.BytesIO):
    # Reject whole-file reads and reads larger than the parser buffer.
    def read(self, size: int | None = -1) -> bytes:
        assert size is not None and 0 <= size <= max(65536, io.DEFAULT_BUFFER_SIZE), "unbounded read"
        return super().read(size)

    # Accept the parser buffer or Python's default buffer, including 128 KiB on 3.14.
    def readinto(self, buffer: Buffer) -> int:
        assert memoryview(buffer).nbytes <= max(65536, io.DEFAULT_BUFFER_SIZE), "unbounded readinto"
        return super().readinto(buffer)


# Priming a cached 2 MB array must read less than a tenth of it and yield row two.
def test_hub_startup_does_not_read_whole_array(monkeypatch: pytest.MonkeyPatch) -> None:
    content = json.dumps([conversation("x" * 2000) for _ in range(1000)]).encode()
    stream = BoundedReadStream(content)
    monkeypatch.setattr(sharegpt, "hf_hub_download", Mock(return_value="cached.json"))
    monkeypatch.setattr(sharegpt, "open", Mock(return_value=stream), raising=False)
    generator = sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(), None)
    assert 0 < stream.tell() < len(content) // 10
    assert next(generator.sharegpt_dataset) == conversation("x" * 2000)


# Two passes over 2,000 rows of 4 KiB each must preserve order and allocate
# less than 2 MiB, catching both whole-array loading and retained records.
def test_memory_stays_bounded_over_two_passes(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    count = 2000
    with path.open("w") as stream:
        stream.write("[")
        for index in range(count):
            stream.write(("," if index else "") + json.dumps(conversation(f"{index}:" + "x" * 4096)))
        stream.write("]")

    tracemalloc.start()
    try:
        generator = sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(path=str(path)), None)
        for index in range(count * 2):
            row = next(generator.sharegpt_dataset)
            assert row["conversations"][0]["value"].startswith(f"{(index + 1) % count}:")
        _, peak = tracemalloc.get_traced_memory()
        # The 8 MB input must never be materialized or retained by repetition.
        assert peak < 2 * 1024 * 1024
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("content", ["[]", " \n[]\n"])
# Empty arrays, with or without whitespace, must fail during priming.
def test_empty_array_fails_without_spinning(content: str, tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="contains no records"):
        sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(path=str(path)), None)


# A directory without JSON files must fail instead of retrying forever.
def test_empty_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no records"):
        sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(path=str(tmp_path)), None)


@pytest.mark.parametrize("content", ["", "[", '[{"conversations": []}', "[{},]", "[{}] trailing"])
# Empty, truncated, and invalid JSON must raise a parser error and close the cached file.
def test_malformed_json_fails_and_closes_stream(content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.BytesIO(content.encode())
    monkeypatch.setattr(sharegpt, "hf_hub_download", Mock(return_value="cached.json"))
    monkeypatch.setattr(sharegpt, "open", Mock(return_value=stream), raising=False)
    with pytest.raises(ijson.JSONError):
        generator = sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(), None)
        next(generator.sharegpt_dataset)
    assert stream.closed


@pytest.mark.parametrize("api", [APIType.Completion, APIType.Chat, APIType.AnthropicMessages])
# Missing or empty conversations are skipped for all APIs; completion token
# limits filter short prompts while chat APIs retain their existing order.
def test_requests_keep_filtering_and_order(api: APIType, tmp_path: Path) -> None:
    rows = [
        conversation("prime"),
        {},
        {"conversations": []},
        {"conversations": None},
        conversation("hello"),
        conversation("bye"),
    ]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    tokenizer = Mock()
    tokenizer.get_tokenizer.return_value.encode.side_effect = lambda text: list(text)
    tokenizer.count_tokens.side_effect = len
    config = DataConfig(
        path=str(path), input_distribution=Distribution(min=4, max=6), output_distribution=Distribution(min=4, max=6)
    )
    generator = sharegpt.HFShareGPTDataGenerator(APIConfig(type=api), config, tokenizer)
    requests = list(itertools.islice(generator.get_data(), 4))
    if api == APIType.Completion:
        assert all(isinstance(request, CompletionAPIData) for request in requests)
        assert [request.prompt for request in requests if isinstance(request, CompletionAPIData)] == [
            "hello",
            "prime",
            "hello",
            "prime",
        ]
        assert all(request.max_tokens == 5 for request in requests if isinstance(request, CompletionAPIData))
    else:
        expected_type = ChatCompletionAPIData if api == APIType.Chat else AnthropicMessagesAPIData
        assert all(isinstance(request, expected_type) for request in requests)
        assert [
            request.messages[0].content
            for request in requests
            if isinstance(request, (ChatCompletionAPIData, AnthropicMessagesAPIData))
        ] == ["hello", "bye", "prime", "hello"]


# Unpickling must not download the Hub file; lazy reload resolves the cache and primes once.
def test_hub_pickle_does_not_reopen_until_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cached.json"
    path.write_text(json.dumps([conversation("first"), conversation("second")]))
    download = Mock(return_value=str(path))
    monkeypatch.setattr(sharegpt, "hf_hub_download", download)
    generator = sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(), None)
    restored = pickle.loads(pickle.dumps(generator))
    assert download.call_count == 1
    restored._ensure_dataset_loaded()
    assert download.call_count == 2
    assert next(restored.sharegpt_dataset) == conversation("second")


# Once initialized, two passes must succeed even if further Hub access would fail.
def test_hub_iteration_only_reads_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [conversation("first"), conversation("second")]
    path = tmp_path / "cached.json"
    path.write_text(json.dumps(rows))
    download = Mock(return_value=str(path))
    monkeypatch.setattr(sharegpt, "hf_hub_download", download)
    generator = sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(), None)
    download.side_effect = AssertionError("Hub accessed after startup")
    assert list(itertools.islice(generator.sharegpt_dataset, 4)) == [rows[1], rows[0]] * 2
    assert download.call_count == 1


# A failed cache download must propagate during initialization before any file read.
def test_hub_download_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sharegpt, "hf_hub_download", Mock(side_effect=OSError("download failed")))
    open_file = Mock()
    monkeypatch.setattr(sharegpt, "open", open_file, raising=False)
    with pytest.raises(OSError, match="download failed"):
        sharegpt.HFShareGPTDataGenerator(APIConfig(), DataConfig(), None)
    open_file.assert_not_called()


@pytest.mark.parametrize("lazy_reload", [False, True])
@pytest.mark.parametrize("stall", ["download", "read"])
# Stalling the download or first cached-file read must time out at startup and
# on lazy reload, without marking a failed reload ready.
def test_deadline_covers_hub_download_and_read(
    lazy_reload: bool, stall: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    finished = threading.Event()
    content = json.dumps([conversation("first"), conversation("second")]).encode()
    path = tmp_path / "cached.json"
    path.write_bytes(content)
    monkeypatch.setattr(sharegpt, "hf_hub_download", Mock(return_value=str(path)))
    config = DataConfig(load_timeout=0.05)
    restored = None
    if lazy_reload:
        generator = sharegpt.HFShareGPTDataGenerator(APIConfig(), config, None)
        restored = pickle.loads(pickle.dumps(generator))

    class SlowStream(io.BytesIO):
        # Hold the first cached-file read until cleanup releases it.
        def readinto(self, buffer: Buffer) -> int:
            try:
                release.wait(5)
                return super().readinto(buffer)
            finally:
                finished.set()

    # Stall the cache download until cleanup releases it, then return its local path.
    def download(*args: object, **kwargs: object) -> str:
        try:
            release.wait(5)
            return str(path)
        finally:
            finished.set()

    if stall == "download":
        monkeypatch.setattr(sharegpt, "hf_hub_download", download)
    else:
        monkeypatch.setattr(sharegpt, "open", lambda *args, **kwargs: SlowStream(content), raising=False)
    try:
        with pytest.raises(TimeoutError, match="data.load_timeout"):
            if restored is None:
                sharegpt.HFShareGPTDataGenerator(APIConfig(), config, None)
            else:
                restored._ensure_dataset_loaded()
        if restored is not None:
            assert not restored._dataset_ready
    finally:
        release.set()
        assert finished.wait(2)
