# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for reading replay traces from S3-compatible object storage.

Everything runs against FakeS3, an in-memory stand-in for a boto3 S3 client that
records each GetObject and the bytes it served, so download counts are asserted
rather than assumed.
"""

import glob
import hashlib
import io
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from inference_perf.datagen.replay.otel_trace_replay_datagen import OTelTraceReplayDataGenerator
from inference_perf.datagen.replay.trace_source import LocalTraceSource
from inference_perf.datagen.replay.remote_trace_store import (
    GetObjectOutput,
    ListPage,
    RemoteObject,
    RemoteReadError,
    RemoteTraceStore,
    is_remote,
    match_key,
    parse_s3_uri,
)

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

BUCKET = "bkt"


class FakeBody:
    def __init__(self, data: bytes, owner: "FakeS3") -> None:
        self._buf = io.BytesIO(data)
        self._owner = owner

    def read(self, amt: Optional[int] = None) -> bytes:
        chunk = self._buf.read(-1 if amt is None else amt)
        self._owner.bytes_served += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class FakeS3:
    """Minimal list_objects_v2 paginator + get_object with IfMatch, in S3 key order."""

    def __init__(self, objects: Dict[str, bytes], page_size: int = 2) -> None:
        self.objects = dict(objects)
        self.page_size = page_size
        self.gets: List[str] = []
        self.bytes_served = 0

    def etag(self, key: str) -> str:
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'

    def get_paginator(self, name: str) -> "FakeS3":
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str = "") -> Iterator[ListPage]:
        assert Bucket == BUCKET
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        for i in range(0, len(keys), self.page_size):
            page = keys[i : i + self.page_size]
            yield {"Contents": [{"Key": k, "ETag": self.etag(k), "Size": len(self.objects[k])} for k in page]}
        if not keys:
            yield {}

    def get_object(self, Bucket: str, Key: str, IfMatch: Optional[str] = None) -> GetObjectOutput:
        self.gets.append(Key)
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        if IfMatch is not None and IfMatch != self.etag(Key):
            raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "etag"}}, "GetObject")
        return {"Body": FakeBody(self.objects[Key], self)}


def wire_record(i: int, session: str = "") -> Dict[str, object]:
    headers = {"session_id": session} if session else {}
    return {
        "path": "/v1/chat/completions",
        "wire": "openai",
        "start_unix": 1.0 + i,
        "headers": headers,
        "request": json.dumps({"model": "m", "messages": [{"role": "user", "content": f"hi {i}"}]}),
        "model": "m",
        "status": 200,
        "response": json.dumps({"choices": [{"message": {"role": "assistant", "content": f"ok {i}"}}]}),
        "total_ms": 5.0,
    }


def wire_bytes(n: int = 2, session: str = "", compress: bool = True) -> bytes:
    raw = "".join(json.dumps(wire_record(i, session)) + "\n" for i in range(n)).encode()
    return zstd.compress(raw) if compress else raw


def make_store(tmp_path: Path, fake: FakeS3, max_bytes: int = 10**9) -> RemoteTraceStore:
    return RemoteTraceStore(tmp_path / "cache", max_bytes, client_factory=lambda: fake)


def obj(fake: FakeS3, key: str) -> RemoteObject:
    return RemoteObject(BUCKET, key, fake.etag(key), len(fake.objects[key]))


# --------------------------------------------------------------------------------------
# URIs and wildcard matching
# --------------------------------------------------------------------------------------


def test_is_remote_and_parse() -> None:
    assert is_remote("s3://b/k") and not is_remote("/local/s3://x") and not is_remote("rel/path")
    assert parse_s3_uri("s3://b/a/b.json") == ("b", "a/b.json")
    assert parse_s3_uri("s3://b") == ("b", "")
    with pytest.raises(ValueError, match="bucket"):
        parse_s3_uri("s3:///key")


@pytest.mark.parametrize(
    "pattern,key,expected",
    [
        ("a/*/c.json", "a/b/c.json", True),
        ("a/*/c.json", "a/b/x/c.json", False),
        ("a/**/c.json", "a/c.json", True),
        ("a/**/c.json", "a/b/x/c.json", True),
        ("a/**", "a/b/x/c.json", True),
        ("a/?.json", "a/b.json", True),
        ("a/[bc].json", "a/c.json", True),
        ("a/*.json", "a/.hidden.json", False),
        ("a/.*.json", "a/.hidden.json", True),
        ("a/**/c.json", "a/.git/c.json", False),
    ],
)
def test_match_key_cases(pattern: str, key: str, expected: bool) -> None:
    assert match_key(pattern, key) is expected


_SEGMENTS = st.sampled_from(["a", "b", "ab", ".h"])
_FILES = st.lists(st.lists(_SEGMENTS, min_size=1, max_size=3), min_size=1, max_size=6)
_PATTERN_SEGMENTS = st.sampled_from(["a", "b", "*", "**", "?", "a*", "[ab]", ".*", "*b"])


@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(files=_FILES, pattern=st.lists(_PATTERN_SEGMENTS, min_size=1, max_size=3))
def test_match_key_parity_with_glob(
    tmp_path_factory: pytest.TempPathFactory, files: List[List[str]], pattern: List[str]
) -> None:
    root = tmp_path_factory.mktemp("tree")
    keys = set()
    for segments in files:
        # A file name is also a directory name elsewhere in the tree only when the shapes allow.
        target = root.joinpath(*segments[:-1], segments[-1] + ".f")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")
        except (FileExistsError, NotADirectoryError):
            continue
        keys.add("/".join(segments[:-1] + [segments[-1] + ".f"]))
    pat = "/".join(pattern[:-1] + [pattern[-1] + ".f"]) if pattern[-1] != "**" else "/".join(pattern)
    expected = {
        os.path.relpath(p, root).replace(os.sep, "/") for p in glob.glob(str(root / pat), recursive=True) if os.path.isfile(p)
    }
    assert {k for k in keys if match_key(pat, k)} == expected


# --------------------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------------------


def test_resolve_pattern_paginates_sorts_and_skips_folder_markers(tmp_path: Path) -> None:
    fake = FakeS3(
        {"m/r1/b.jsonl.zst": b"x", "m/r1/a.jsonl.zst": b"y", "m/r2/c.jsonl.zst": b"z", "m/r2/": b"", "n/x.jsonl": b""}
    )
    found = make_store(tmp_path, fake).resolve(f"s3://{BUCKET}/m/*/*.jsonl.zst")
    assert [o.key for o in found] == ["m/r1/a.jsonl.zst", "m/r1/b.jsonl.zst", "m/r2/c.jsonl.zst"]
    assert found[0].uri == f"s3://{BUCKET}/m/r1/a.jsonl.zst" and found[0].name == "a.jsonl.zst"
    assert found[0].etag == fake.etag("m/r1/a.jsonl.zst") and found[0].size == 1
    assert fake.gets == []


def test_resolve_exact_key_and_missing_key(tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": b"1", "d/a.jsonl.zst": b"2"})
    store = make_store(tmp_path, fake)
    assert [o.key for o in store.resolve(f"s3://{BUCKET}/d/a.jsonl")] == ["d/a.jsonl"]
    with pytest.raises(ValueError, match=f"Trace file does not exist: s3://{BUCKET}/d/b.jsonl"):
        store.resolve(f"s3://{BUCKET}/d/b.jsonl")


def test_resolve_pattern_without_matches_is_empty(tmp_path: Path) -> None:
    assert make_store(tmp_path, FakeS3({"d/a.jsonl": b""})).resolve(f"s3://{BUCKET}/zzz/*.jsonl") == []


def test_resolve_without_credentials_names_bucket_and_env_vars(tmp_path: Path) -> None:
    class NoCreds(FakeS3):
        def paginate(self, Bucket: str, Prefix: str = "") -> Iterator[ListPage]:
            raise NoCredentialsError()

    with pytest.raises(ValueError, match=r"s3://bkt.*AWS_ACCESS_KEY_ID.*AWS_ENDPOINT_URL_S3"):
        make_store(tmp_path, NoCreds({})).resolve(f"s3://{BUCKET}/*.jsonl")


# --------------------------------------------------------------------------------------
# Probing the first record
# --------------------------------------------------------------------------------------


def test_first_record_reads_only_the_start_of_a_large_compressed_object(tmp_path: Path) -> None:
    big = [json.dumps(wire_record(0))] + [os.urandom(2048).hex() for _ in range(400)]
    data = zstd.compress(("\n".join(big) + "\n").encode())
    fake = FakeS3({"k/calls.jsonl.zst": data})
    record = make_store(tmp_path, fake).first_record(obj(fake, "k/calls.jsonl.zst"))
    assert record == wire_record(0)
    assert fake.bytes_served < len(data) / 4


def test_first_record_plain_skips_blank_lines(tmp_path: Path) -> None:
    fake = FakeS3({"k/calls.jsonl": b"\n\n" + wire_bytes(compress=False)})
    assert make_store(tmp_path, fake).first_record(obj(fake, "k/calls.jsonl")) == wire_record(0)


def test_first_record_bad_json_is_value_error(tmp_path: Path) -> None:
    fake = FakeS3({"k/calls.jsonl": b"{nope\n"})
    with pytest.raises(ValueError, match="s3://bkt/k/calls.jsonl"):
        make_store(tmp_path, fake).first_record(obj(fake, "k/calls.jsonl"))


def test_first_records_returns_errors_in_place(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": wire_bytes(compress=False), "b.jsonl": b"{nope\n"})
    out = make_store(tmp_path, fake).first_records([obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    assert out[0] == wire_record(0) and isinstance(out[1], ValueError)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def test_materialize_downloads_once_and_keeps_basename(tmp_path: Path) -> None:
    fake = FakeS3({"m/r/task a-calls.jsonl.zst": wire_bytes()})
    store = make_store(tmp_path, fake)
    o = obj(fake, "m/r/task a-calls.jsonl.zst")
    first = store.materialize(o)
    assert first.name == "task a-calls.jsonl.zst" and first.read_bytes() == fake.objects[o.key]
    assert store.materialize(o) == first
    assert fake.gets == [o.key]
    assert store.sync_downloads == 1
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_materialize_refetches_when_etag_changes(tmp_path: Path) -> None:
    fake = FakeS3({"k.jsonl": b"v1"})
    store = make_store(tmp_path, fake)
    store.materialize(obj(fake, "k.jsonl"))
    fake.objects["k.jsonl"] = b"v2"
    assert store.materialize(obj(fake, "k.jsonl")).read_bytes() == b"v2"
    assert len(fake.gets) == 2


def test_object_changed_after_listing_is_a_read_error(tmp_path: Path) -> None:
    fake = FakeS3({"k.jsonl": b"v1"})
    listed = obj(fake, "k.jsonl")
    fake.objects["k.jsonl"] = b"v2"
    store = make_store(tmp_path, fake)
    with pytest.raises(RemoteReadError, match="changed since it was listed"):
        store.materialize(listed)
    with pytest.raises(RemoteReadError, match="changed since it was listed"):
        store.first_record(listed)
    assert isinstance(RemoteReadError("x"), OSError)


def test_missing_object_is_a_read_error(tmp_path: Path) -> None:
    fake = FakeS3({})
    with pytest.raises(RemoteReadError, match="s3://bkt/gone.jsonl"):
        make_store(tmp_path, fake).materialize(RemoteObject(BUCKET, "gone.jsonl", '"e"', 1))


def cached_files(tmp_path: Path) -> List[str]:
    return sorted(p.name for p in (tmp_path / "cache").rglob("*") if p.is_file())


def test_eviction_keeps_cache_under_cap_least_recently_used_first(tmp_path: Path) -> None:
    fake = FakeS3({f"{c}.jsonl": c.encode() * 100 for c in "abc"})
    store = make_store(tmp_path, fake, max_bytes=250)
    a = store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "b.jsonl"))
    os.utime(a)  # a hit touches; touch by hand to make the order explicit
    store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "c.jsonl"))
    assert cached_files(tmp_path) == ["a.jsonl", "c.jsonl"]


def test_object_larger_than_cap_still_loads(tmp_path: Path) -> None:
    fake = FakeS3({"big.jsonl": b"x" * 500, "small.jsonl": b"y"})
    store = make_store(tmp_path, fake, max_bytes=100)
    assert store.materialize(obj(fake, "big.jsonl")).read_bytes() == b"x" * 500
    store.materialize(obj(fake, "small.jsonl"))
    assert cached_files(tmp_path) == ["small.jsonl"]


def test_cap_zero_keeps_only_the_file_just_written(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a", "b.jsonl": b"b"})
    store = make_store(tmp_path, fake, max_bytes=0)
    assert store.materialize(obj(fake, "a.jsonl")).read_bytes() == b"a"
    assert store.materialize(obj(fake, "b.jsonl")).read_bytes() == b"b"
    assert cached_files(tmp_path) == ["b.jsonl"]


# --------------------------------------------------------------------------------------
# Prefetch and pickling
# --------------------------------------------------------------------------------------


def wait_until_cached(store: RemoteTraceStore, objs: List[RemoteObject], timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while not all(store.cache_path(o).exists() for o in objs) or store._inflight:
        assert time.monotonic() < deadline, "prefetch did not finish"
        time.sleep(0.01)


def test_prefetch_then_materialize_is_not_a_synchronous_download(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a", "b.jsonl": b"b"})
    store = make_store(tmp_path, fake)
    store.prefetch([obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    wait_until_cached(store, [obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "b.jsonl"))
    assert sorted(fake.gets) == ["a.jsonl", "b.jsonl"]
    assert store.sync_downloads == 0


def test_download_all_fills_the_cache(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a", "b.jsonl": b"b"})
    store = make_store(tmp_path, fake)
    store.download_all([obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    assert cached_files(tmp_path) == ["a.jsonl", "b.jsonl"]
    store.materialize(obj(fake, "a.jsonl"))
    assert store.sync_downloads == 0 and len(fake.gets) == 2


_SHARED: Dict[str, FakeS3] = {}


def _shared_client() -> FakeS3:
    return _SHARED["fake"]


def test_unpickled_store_has_no_client_and_never_prefetches(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a"})
    _SHARED["fake"] = fake
    store = RemoteTraceStore(tmp_path / "cache", 10**6, client_factory=_shared_client)
    store.materialize(obj(fake, "a.jsonl"))  # creates the client
    worker = pickle.loads(pickle.dumps(store))
    assert worker.prefetch_enabled is False and store.prefetch_enabled is True
    worker.prefetch([RemoteObject(BUCKET, "never.jsonl", '"e"', 1)])
    assert "never.jsonl" not in fake.gets
    assert worker.materialize(obj(fake, "a.jsonl")).read_bytes() == b"a"  # cache hit, no download
    assert fake.gets == ["a.jsonl"]


def test_waiting_on_an_unfinished_prefetch_is_counted(tmp_path: Path) -> None:
    import threading

    release = threading.Event()

    class Slow(FakeS3):
        def get_object(self, Bucket: str, Key: str, IfMatch: Optional[str] = None) -> GetObjectOutput:
            release.wait(5)
            return super().get_object(Bucket, Key, IfMatch)

    fake = Slow({"a.jsonl": b"a"})
    store = make_store(tmp_path, fake)
    store.prefetch([obj(fake, "a.jsonl")])
    threading.Timer(0.2, release.set).start()
    assert store.materialize(obj(fake, "a.jsonl")).read_bytes() == b"a"
    assert fake.gets == ["a.jsonl"]
    assert store.sync_downloads == 1 and store.sync_wait_s >= 0.1


# --------------------------------------------------------------------------------------
# LocalTraceSource over remote objects
# --------------------------------------------------------------------------------------


def remote_source(
    store: RemoteTraceStore, entries: Sequence[Union[Path, RemoteObject]], skip_invalid: bool = False
) -> LocalTraceSource:
    from inference_perf.datagen.replay.otel_trace_replay_datagen import _normalize_file_trace

    return LocalTraceSource(entries, skip_invalid, _normalize_file_trace, remote=store)


def test_remote_wire_is_indexed_by_probe_and_loaded_from_cache(tmp_path: Path) -> None:
    fake = FakeS3({"m/r/a-calls.jsonl.zst": wire_bytes(3), "m/r/b-calls.jsonl.zst": wire_bytes(2, session="sess-1")})
    store = make_store(tmp_path, fake)
    objs = store.resolve(f"s3://{BUCKET}/m/*/*.jsonl.zst")
    source = remote_source(store, objs)
    records = source.list_records()
    uri_a = f"s3://{BUCKET}/m/r/a-calls.jsonl.zst"
    assert [r.source_id for r in records] == [uri_a, f"s3://{BUCKET}/m/r/b-calls.jsonl.zst"]
    assert records[0].session_id_suffix == hashlib.sha256(uri_a.encode()).hexdigest()[:32]
    assert records[1].session_id_suffix == "sess1"
    assert sorted(fake.gets) == sorted(o.key for o in objs)  # one probe each, no full download
    assert not list((tmp_path / "cache").rglob("*.zst"))
    loaded = source.load_record(0)
    assert len(loaded["spans"]) == 3 and loaded["source_id"] == uri_a
    assert loaded["trace_id"] == records[0].session_id_suffix
    source.load_record(0)
    assert fake.gets.count("m/r/a-calls.jsonl.zst") == 2  # probe + one download


def test_remote_ids_do_not_depend_on_the_cache_dir(tmp_path: Path) -> None:
    fake = FakeS3({"k/a.jsonl.zst": wire_bytes()})
    ids = []
    for sub in ("one", "two"):
        store = RemoteTraceStore(tmp_path / sub, 10**9, client_factory=lambda: fake)
        source = remote_source(store, store.resolve(f"s3://{BUCKET}/k/*.jsonl.zst"))
        ids.append((source.list_records(), source.load_record(0)["spans"][0]["trace_id"]))
    assert ids[0] == ids[1]


def otel_doc(trace_id: str) -> Dict[str, object]:
    return {"trace_id": trace_id, "spans": []}


def test_remote_otel_jsonl_rows_are_addressed_by_uri(tmp_path: Path) -> None:
    body = (json.dumps(otel_doc("t0")) + "\n" + json.dumps(otel_doc("t1")) + "\n").encode()
    fake = FakeS3({"d/traces.jsonl": body})
    store = make_store(tmp_path, fake)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.jsonl"))
    uri = f"s3://{BUCKET}/d/traces.jsonl"
    assert [(r.source_id, r.session_id_suffix) for r in source.list_records()] == [(f"{uri}#0", "t0"), (f"{uri}#1", "t1")]
    assert source.load_record(1)["trace_id"] == "t1"


def test_remote_otel_json_document(tmp_path: Path) -> None:
    fake = FakeS3({"d/one.json": json.dumps(otel_doc("solo"), indent=2).encode()})
    store = make_store(tmp_path, fake)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.json"))
    [record] = source.list_records()
    assert (record.source_id, record.session_id_suffix) == (f"s3://{BUCKET}/d/one.json", "solo")
    assert source.load_record(0)["source_id"] == f"s3://{BUCKET}/d/one.json"


def test_broken_remote_wire_surfaces_at_load_naming_the_uri(tmp_path: Path) -> None:
    from inference_perf.datagen.replay.trace_source import InvalidTraceError

    good = json.dumps(wire_record(0)) + "\n{broken\n"
    fake = FakeS3({"k/bad.jsonl": good.encode()})
    store = make_store(tmp_path, fake)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/k/*.jsonl"))
    assert len(source.list_records()) == 1  # the probe only sees line 1
    with pytest.raises(InvalidTraceError, match=f"s3://{BUCKET}/k/bad.jsonl"):
        source.load_record(0)


def test_unreadable_remote_object_is_skipped_or_fatal(tmp_path: Path) -> None:
    from inference_perf.datagen.replay.trace_source import TraceReadError

    fake = FakeS3({"k/a.jsonl": wire_bytes(compress=False), "k/b.jsonl": wire_bytes(compress=False)})
    store = make_store(tmp_path, fake)
    objs = store.resolve(f"s3://{BUCKET}/k/*.jsonl")
    fake.objects["k/a.jsonl"] += b"\n"  # changed after listing
    assert [r.source_id for r in remote_source(store, objs, skip_invalid=True).list_records()] == [objs[1].uri]
    with pytest.raises(TraceReadError, match="changed since it was listed"):
        remote_source(store, objs, skip_invalid=False)


def test_local_and_remote_entries_mix(tmp_path: Path) -> None:
    local = tmp_path / "local-calls.jsonl"
    local.write_bytes(wire_bytes(compress=False))
    fake = FakeS3({"k/r.jsonl": wire_bytes(compress=False)})
    store = make_store(tmp_path, fake)
    source = remote_source(store, [local, *store.resolve(f"s3://{BUCKET}/k/*.jsonl")])
    assert [r.source_id for r in source.list_records()] == [str(local), f"s3://{BUCKET}/k/r.jsonl"]
    assert len(source.load_record(0)["spans"]) == 2 and len(source.load_record(1)["spans"]) == 2


# --------------------------------------------------------------------------------------
# OTelTraceReplayDataGenerator over s3:// entries
# --------------------------------------------------------------------------------------


def make_generator(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeS3,
    tmp_path: Path,
    trace_files: Optional[List[str]] = None,
    trace_directory: Optional[str] = None,
    remote_cache_max_gb: float = 10.0,
) -> OTelTraceReplayDataGenerator:
    import inference_perf.datagen.replay.remote_trace_store as remote_module
    from inference_perf.config import APIConfig, APIType, DataConfig, OTelTraceReplayConfig

    _SHARED["fake"] = fake
    monkeypatch.setattr(remote_module, "default_client", _shared_client)
    replay = OTelTraceReplayConfig(
        trace_files=trace_files,
        trace_directory=trace_directory,
        remote_cache_dir=str(tmp_path / "cache"),
        remote_cache_max_gb=remote_cache_max_gb,
    )
    config = DataConfig(type="otel_trace_replay", otel_trace_replay=replay)
    return OTelTraceReplayDataGenerator(
        api_config=APIConfig(type=APIType.Chat, streaming=False), config=config, tokenizer=None, num_workers=1
    )


def remote_parts(gen: OTelTraceReplayDataGenerator) -> Tuple[RemoteTraceStore, LocalTraceSource]:
    assert gen._remote_store is not None and isinstance(gen._trace_source, LocalTraceSource)
    return gen._remote_store, gen._trace_source


def corpus(n: int) -> FakeS3:
    return FakeS3({f"m/run/task{i}-calls.jsonl.zst": wire_bytes(2, session=f"s{i}") for i in range(n)})


def test_generator_replays_remote_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gen = make_generator(monkeypatch, corpus(3), tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*-calls.jsonl.zst"])
    assert gen.get_session_count() == 3
    for i in range(3):
        assert gen.is_session_buildable(i)
    assert {s.source_id for s in gen.sessions if s is not None} == {
        f"s3://{BUCKET}/m/run/task{i}-calls.jsonl.zst" for i in range(3)
    }


def test_each_object_is_downloaded_once_across_parent_worker_and_cycling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = corpus(3)
    gen = make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/**/*.jsonl.zst"])
    # Fits the cache: probe + pre-download, before any session is built.
    assert sorted(fake.gets) == sorted(list(fake.objects) * 2)
    for i in range(3):
        gen.is_session_buildable(i)
    worker = pickle.loads(pickle.dumps(gen))
    for i in range(3):
        worker._ensure_session_built(i)
    gen._ensure_session_built(3)  # corpus cycling replays slot 0
    assert len(fake.gets) == 6
    assert remote_parts(gen)[0].sync_downloads == 0 and remote_parts(worker)[0].sync_downloads == 0


def test_set_larger_than_the_cache_is_downloaded_ahead_in_play_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = corpus(6)
    size = len(next(iter(fake.objects.values())))
    gen = make_generator(
        monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_max_gb=(4.5 * size) / 1e9
    )
    store, source = remote_parts(gen)
    play = [source.records[row].source_id for row in gen._row_order]
    by_uri = {r.remote.uri: r.remote for r in source.records if r.remote is not None}
    cached = [uri for uri in play if store.cache_path(by_uri[uri]).exists()]
    # Warm-up covers half the cap: the first two sessions in play order, not more.
    assert cached == play[:2]
    for i in range(6):
        gen.is_session_buildable(i)
        wait_until_cached(store, [])
    assert store.sync_downloads <= 1  # at most the one session the window could not cover
    assert sum(p.stat().st_size for p in (tmp_path / "cache").rglob("*.zst")) <= 4.5 * size + size


def test_worker_copy_never_prefetches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = corpus(6)
    size = len(next(iter(fake.objects.values())))
    gen = make_generator(
        monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_max_gb=(3.5 * size) / 1e9
    )
    worker = pickle.loads(pickle.dumps(gen))
    before = len(fake.gets)
    worker._ensure_session_built(0)  # already warmed: a cache hit, and no prefetch of what follows
    assert len(fake.gets) == before


def test_remote_trace_directory_lists_direct_children_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": wire_bytes(compress=False), "d/sub/b.jsonl": wire_bytes(compress=False), "d/notes.txt": b"x"})
    gen = make_generator(monkeypatch, fake, tmp_path, trace_directory=f"s3://{BUCKET}/d/")
    assert gen.get_session_count() == 1


def test_remote_match_without_trace_suffix_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": wire_bytes(compress=False), "d/notes.txt": b"x"})
    with pytest.raises(ValueError, match=r"must be a \.json or \.jsonl file.*s3://bkt/d/notes\.txt"):
        make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/d/*"])


def test_remote_compressed_duplicate_is_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": wire_bytes(compress=False), "d/a.jsonl.zst": wire_bytes()})
    gen = make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/d/*.jsonl*"])
    assert gen.get_session_count() == 1


def test_remote_pattern_with_no_matches_fails_like_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No trace files resolved"):
        make_generator(monkeypatch, corpus(1), tmp_path, trace_files=[f"s3://{BUCKET}/nope/*.jsonl.zst"])


def test_config_rejects_s3_uri_without_bucket() -> None:
    from pydantic import ValidationError

    from inference_perf.config import OTelTraceReplayConfig

    with pytest.raises(ValidationError, match="missing a bucket"):
        OTelTraceReplayConfig(trace_files=["s3:///k/*.jsonl"])
    with pytest.raises(ValidationError, match="missing a bucket"):
        OTelTraceReplayConfig(trace_directory="s3://")
    assert OTelTraceReplayConfig(trace_files=["s3://b/k/*.jsonl"]).remote_cache_max_gb == 10.0
