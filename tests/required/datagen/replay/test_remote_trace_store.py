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

from inference_perf.datagen.replay.otel_trace_replay_datagen import OTelTraceReplayDataGenerator
from inference_perf.datagen.replay.trace_source import LocalTraceRecord, LocalTraceSource
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


# A GetObject body over fixed bytes; every byte read is added to the owner's bytes_served.
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


# An in-memory S3 client: paginated listing (Prefix, Delimiter), GetObject with IfMatch and
# Range, and a record of every key fetched, range asked for and prefix listed.
class FakeS3:
    """Minimal list_objects_v2 paginator + get_object with IfMatch, in S3 key order."""

    def __init__(self, objects: Dict[str, bytes], page_size: int = 2) -> None:
        self.objects = dict(objects)
        self.page_size = page_size
        self.gets: List[str] = []
        self.ranges: List[Optional[str]] = []
        self.listed: List[Tuple[str, str]] = []
        self.bytes_served = 0

    def etag(self, key: str) -> str:
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'

    def get_paginator(self, name: str) -> "FakeS3":
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str = "", Delimiter: str = "") -> Iterator[ListPage]:
        assert Bucket == BUCKET
        self.listed.append((Prefix, Delimiter))
        keys = sorted(k for k in self.objects if k.startswith(Prefix) and not (Delimiter and Delimiter in k[len(Prefix) :]))
        for i in range(0, len(keys), self.page_size):
            page = keys[i : i + self.page_size]
            yield {"Contents": [{"Key": k, "ETag": self.etag(k), "Size": len(self.objects[k])} for k in page]}
        if not keys:
            yield {}

    def get_object(self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None) -> GetObjectOutput:
        self.gets.append(Key)
        self.ranges.append(Range)
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        if IfMatch is not None and IfMatch != self.etag(Key):
            raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "etag"}}, "GetObject")
        data = self.objects[Key]
        if Range is not None:
            start, end = Range.removeprefix("bytes=").split("-")
            if int(start) >= len(data):  # what S3 answers for a range of an empty object
                raise ClientError({"Error": {"Code": "InvalidRange", "Message": "416"}}, "GetObject")
            data = data[int(start) : int(end) + 1]
        return {"Body": FakeBody(data, self)}


# One Chat Completions wire record for call i: wire_record(0) asks "hi 0" and answers "ok 0";
# session sets the session_id header.
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


# n wire records as JSON lines, zstd-compressed unless compress=False.
def wire_bytes(n: int = 2, session: str = "", compress: bool = True) -> bytes:
    raw = "".join(json.dumps(wire_record(i, session)) + "\n" for i in range(n)).encode()
    return zstd.compress(raw) if compress else raw


# A store over fake with its cache under tmp_path/cache and a max_bytes budget (default 1 GB).
def make_store(tmp_path: Path, fake: FakeS3, max_bytes: int = 10**9) -> RemoteTraceStore:
    return RemoteTraceStore(tmp_path / "cache", max_bytes, client_factory=lambda: fake)


# The listed RemoteObject for key in fake, with its current ETag and size.
def obj(fake: FakeS3, key: str) -> RemoteObject:
    return RemoteObject(BUCKET, key, fake.etag(key), len(fake.objects[key]))


# --------------------------------------------------------------------------------------
# URIs and wildcard matching
# --------------------------------------------------------------------------------------


# "s3://b/a/b.json" splits into ("b", "a/b.json"), "s3://b" into ("b", ""); "s3:///key" has no
# bucket and raises. Local paths, even ones containing s3://, are not remote.
def test_is_remote_and_parse() -> None:
    assert is_remote("s3://b/k") and not is_remote("/local/s3://x") and not is_remote("rel/path")
    assert parse_s3_uri("s3://b/a/b.json") == ("b", "a/b.json")
    assert parse_s3_uri("s3://b") == ("b", "")
    with pytest.raises(ValueError, match="bucket"):
        parse_s3_uri("s3:///key")


# Each (pattern, key) matches exactly when glob.glob(recursive=True) would: * stays in one
# folder, ** spans folders but not hidden ones, a leading * never matches a dotfile.
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


_SEGMENTS = ["a", "b", "ab", ".h"]
_PATTERN_SEGMENTS = ["a", "b", "*", "**", "?", "a*", "[ab]", ".*", "*b"]


# 400 seeded random file trees and patterns (segments a, b, ab, .h; *, **, ?, [ab], .*):
# match_key selects exactly the files glob.glob(recursive=True) returns.
@pytest.mark.parametrize("seed", range(400))
def test_match_key_parity_with_glob(tmp_path: Path, seed: int) -> None:
    import random

    rng = random.Random(seed)
    files = [[rng.choice(_SEGMENTS) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(1, 6))]
    pattern = [rng.choice(_PATTERN_SEGMENTS) for _ in range(rng.randint(1, 3))]
    root = tmp_path / "tree"
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


# m/*/*.jsonl.zst over 2-key pages returns the three matching keys sorted, skips the folder
# marker m/r2/, carries each ETag and size, and fetches nothing.
def test_resolve_pattern_paginates_sorts_and_skips_folder_markers(tmp_path: Path) -> None:
    fake = FakeS3(
        {"m/r1/b.jsonl.zst": b"x", "m/r1/a.jsonl.zst": b"y", "m/r2/c.jsonl.zst": b"z", "m/r2/": b"", "n/x.jsonl": b""}
    )
    found = make_store(tmp_path, fake).resolve(f"s3://{BUCKET}/m/*/*.jsonl.zst")
    assert [o.key for o in found] == ["m/r1/a.jsonl.zst", "m/r1/b.jsonl.zst", "m/r2/c.jsonl.zst"]
    assert found[0].uri == f"s3://{BUCKET}/m/r1/a.jsonl.zst" and found[0].name == "a.jsonl.zst"
    assert found[0].etag == fake.etag("m/r1/a.jsonl.zst") and found[0].size == 1
    assert fake.gets == []


# An exact key that exists resolves to itself (not its .zst sibling); a missing exact key
# raises "Trace file does not exist", as for a local file.
def test_resolve_exact_key_and_missing_key(tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": b"1", "d/a.jsonl.zst": b"2"})
    store = make_store(tmp_path, fake)
    assert [o.key for o in store.resolve(f"s3://{BUCKET}/d/a.jsonl")] == ["d/a.jsonl"]
    with pytest.raises(ValueError, match=f"Trace file does not exist: s3://{BUCKET}/d/b.jsonl"):
        store.resolve(f"s3://{BUCKET}/d/b.jsonl")


# A pattern matching nothing (zzz/*.jsonl) resolves to [] rather than raising.
def test_resolve_pattern_without_matches_is_empty(tmp_path: Path) -> None:
    assert make_store(tmp_path, FakeS3({"d/a.jsonl": b""})).resolve(f"s3://{BUCKET}/zzz/*.jsonl") == []


# Listing with no credentials raises a ValueError naming s3://bkt and AWS_ACCESS_KEY_ID /
# AWS_ENDPOINT_URL_S3.
def test_resolve_without_credentials_names_bucket_and_env_vars(tmp_path: Path) -> None:
    class NoCreds(FakeS3):
        def paginate(self, Bucket: str, Prefix: str = "", Delimiter: str = "") -> Iterator[ListPage]:
            raise NoCredentialsError()

    with pytest.raises(ValueError, match=r"s3://bkt.*AWS_ACCESS_KEY_ID.*AWS_ENDPOINT_URL_S3"):
        make_store(tmp_path, NoCreds({})).resolve(f"s3://{BUCKET}/*.jsonl")


# --------------------------------------------------------------------------------------
# Probing the first record
# --------------------------------------------------------------------------------------


# A ~4 MB .zst whose first line is a wire record: the probe returns that record after reading
# under a quarter of the object.
def test_first_record_reads_only_the_start_of_a_large_compressed_object(tmp_path: Path) -> None:
    big = [json.dumps(wire_record(0))] + [os.urandom(2048).hex() for _ in range(2000)]
    data = zstd.compress(("\n".join(big) + "\n").encode())
    fake = FakeS3({"k/calls.jsonl.zst": data})
    record = make_store(tmp_path, fake).first_record(obj(fake, "k/calls.jsonl.zst"))
    assert record == wire_record(0)
    assert fake.bytes_served < len(data) / 4


# Two blank lines before the first record of a plain .jsonl: the probe skips them and returns it.
def test_first_record_plain_skips_blank_lines(tmp_path: Path) -> None:
    fake = FakeS3({"k/calls.jsonl": b"\n\n" + wire_bytes(compress=False)})
    assert make_store(tmp_path, fake).first_record(obj(fake, "k/calls.jsonl")) == wire_record(0)


# A first line "{nope" is a ValueError naming s3://bkt/k/calls.jsonl.
def test_first_record_bad_json_is_value_error(tmp_path: Path) -> None:
    fake = FakeS3({"k/calls.jsonl": b"{nope\n"})
    with pytest.raises(ValueError, match="s3://bkt/k/calls.jsonl"):
        make_store(tmp_path, fake).first_record(obj(fake, "k/calls.jsonl"))


# Probing [good, bad JSON] returns [record, ValueError] in the same order, without raising.
def test_first_records_returns_errors_in_place(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": wire_bytes(compress=False), "b.jsonl": b"{nope\n"})
    out = make_store(tmp_path, fake).first_records([obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    assert out[0] == wire_record(0) and isinstance(out[1], ValueError)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


# A key with a space downloads once into a file with the same basename; the second call is a
# cache hit, one wait is counted, and no .part file is left.
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


# The object changes (v1 -> v2, new ETag): materialize downloads the new version.
def test_materialize_refetches_when_etag_changes(tmp_path: Path) -> None:
    fake = FakeS3({"k.jsonl": b"v1"})
    store = make_store(tmp_path, fake)
    store.materialize(obj(fake, "k.jsonl"))
    fake.objects["k.jsonl"] = b"v2"
    assert store.materialize(obj(fake, "k.jsonl")).read_bytes() == b"v2"
    assert len(fake.gets) == 2


# Listed at v1, now v2: both download and probe fail with "changed since it was listed".
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


# A listed object that is gone fails with a RemoteReadError naming its URI.
def test_missing_object_is_a_read_error(tmp_path: Path) -> None:
    fake = FakeS3({})
    with pytest.raises(RemoteReadError, match="s3://bkt/gone.jsonl"):
        make_store(tmp_path, fake).materialize(RemoteObject(BUCKET, "gone.jsonl", '"e"', 1))


# The basenames of the complete files in the cache, sorted.
def cached_files(tmp_path: Path) -> List[str]:
    return sorted(p.name for p in (tmp_path / "cache").rglob("*") if p.is_file())


# Budget 250 B, three 100 B objects, a used after b: downloading c evicts b and keeps a, c.
def test_eviction_keeps_cache_under_cap_least_recently_used_first(tmp_path: Path) -> None:
    fake = FakeS3({f"{c}.jsonl": c.encode() * 100 for c in "abc"})
    store = make_store(tmp_path, fake, max_bytes=250)
    a = store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "b.jsonl"))
    os.utime(a)  # a hit touches; touch by hand to make the order explicit
    store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "c.jsonl"))
    assert cached_files(tmp_path) == ["a.jsonl", "c.jsonl"]


# A 500 B object with a 100 B budget still loads; the next download evicts it.
def test_object_larger_than_cap_still_loads(tmp_path: Path) -> None:
    fake = FakeS3({"big.jsonl": b"x" * 500, "small.jsonl": b"y"})
    store = make_store(tmp_path, fake, max_bytes=100)
    assert store.materialize(obj(fake, "big.jsonl")).read_bytes() == b"x" * 500
    store.materialize(obj(fake, "small.jsonl"))
    assert cached_files(tmp_path) == ["small.jsonl"]


# Budget 0: each download loads, and only the most recent file stays on disk.
def test_cap_zero_keeps_only_the_file_just_written(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a", "b.jsonl": b"b"})
    store = make_store(tmp_path, fake, max_bytes=0)
    assert store.materialize(obj(fake, "a.jsonl")).read_bytes() == b"a"
    assert store.materialize(obj(fake, "b.jsonl")).read_bytes() == b"b"
    assert cached_files(tmp_path) == ["b.jsonl"]


# --------------------------------------------------------------------------------------
# Prefetch and pickling
# --------------------------------------------------------------------------------------


# Wait (up to 5 s) until objs are cached and no background download is in flight.
def wait_until_cached(store: RemoteTraceStore, objs: List[RemoteObject], timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while not all(store.cache_path(o).exists() for o in objs) or store._inflight:
        assert time.monotonic() < deadline, "prefetch did not finish"
        time.sleep(0.01)


# Prefetch a and b, let them finish: reading them is two cache hits and zero waits.
def test_prefetch_then_materialize_is_not_a_synchronous_download(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a", "b.jsonl": b"b"})
    store = make_store(tmp_path, fake)
    store.prefetch([obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    wait_until_cached(store, [obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "b.jsonl"))
    assert sorted(fake.gets) == ["a.jsonl", "b.jsonl"]
    assert store.sync_downloads == 0


# download_all([a, b]) caches both; reading a afterwards is a hit with no wait and no GET.
def test_download_all_fills_the_cache(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a", "b.jsonl": b"b"})
    store = make_store(tmp_path, fake)
    store.download_all([obj(fake, "a.jsonl"), obj(fake, "b.jsonl")])
    assert cached_files(tmp_path) == ["a.jsonl", "b.jsonl"]
    store.materialize(obj(fake, "a.jsonl"))
    assert store.sync_downloads == 0 and len(fake.gets) == 2


_SHARED: Dict[str, FakeS3] = {}


# A module-level client factory, so a store that uses it can be pickled.
def _shared_client() -> FakeS3:
    return _SHARED["fake"]


# A pickled copy (as a worker gets) has prefetch off, ignores prefetch requests, and still reads
# the parent's cached a without downloading it again.
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


# A read that waits 0.2 s for its still-running prefetch is counted as one wait of at least 0.1 s,
# and the object is fetched once.
def test_waiting_on_an_unfinished_prefetch_is_counted(tmp_path: Path) -> None:
    import threading

    release = threading.Event()

    class Slow(FakeS3):
        def get_object(
            self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None
        ) -> GetObjectOutput:
            release.wait(5)
            return super().get_object(Bucket, Key, IfMatch, Range)

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


# A LocalTraceSource over entries (local paths or RemoteObjects) reading through store.
def remote_source(
    store: RemoteTraceStore, entries: Sequence[Union[Path, RemoteObject]], skip_invalid: bool = False
) -> LocalTraceSource:
    from inference_perf.datagen.replay.otel_trace_replay_datagen import _normalize_file_trace

    return LocalTraceSource(entries, skip_invalid, _normalize_file_trace, remote=store)


# Two wire captures: indexing probes each once and caches nothing; ids are sha256(uri)[:32] or
# the session header; loading downloads once and reuses the cached copy.
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


# The same object read through caches one/ and two/ gets the same records and trace id.
def test_remote_ids_do_not_depend_on_the_cache_dir(tmp_path: Path) -> None:
    fake = FakeS3({"k/a.jsonl.zst": wire_bytes()})
    ids = []
    for sub in ("one", "two"):
        store = RemoteTraceStore(tmp_path / sub, 10**9, client_factory=lambda: fake)
        source = remote_source(store, store.resolve(f"s3://{BUCKET}/k/*.jsonl.zst"))
        ids.append((source.list_records(), source.load_record(0)["spans"][0]["trace_id"]))
    assert ids[0] == ids[1]


# A minimal OTel trace document with this trace_id and no spans.
def otel_doc(trace_id: str) -> Dict[str, object]:
    return {"trace_id": trace_id, "spans": []}


# A two-row OTel .jsonl gives records uri#0 / uri#1 with session ids t0 / t1; row 1 loads as t1.
def test_remote_otel_jsonl_rows_are_addressed_by_uri(tmp_path: Path) -> None:
    body = (json.dumps(otel_doc("t0")) + "\n" + json.dumps(otel_doc("t1")) + "\n").encode()
    fake = FakeS3({"d/traces.jsonl": body})
    store = make_store(tmp_path, fake)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.jsonl"))
    uri = f"s3://{BUCKET}/d/traces.jsonl"
    assert [(r.source_id, r.session_id_suffix) for r in source.list_records()] == [(f"{uri}#0", "t0"), (f"{uri}#1", "t1")]
    assert source.load_record(1)["trace_id"] == "t1"


# A pretty-printed OTel .json: one record with source_id = its URI and session id = its trace_id.
def test_remote_otel_json_document(tmp_path: Path) -> None:
    fake = FakeS3({"d/one.json": json.dumps(otel_doc("solo"), indent=2).encode()})
    store = make_store(tmp_path, fake)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.json"))
    [record] = source.list_records()
    assert (record.source_id, record.session_id_suffix) == (f"s3://{BUCKET}/d/one.json", "solo")
    assert source.load_record(0)["source_id"] == f"s3://{BUCKET}/d/one.json"


# A capture whose second line is broken indexes from line 1, then fails at load with an
# InvalidTraceError naming its URI.
def test_broken_remote_wire_surfaces_at_load_naming_the_uri(tmp_path: Path) -> None:
    from inference_perf.datagen.replay.trace_source import InvalidTraceError

    good = json.dumps(wire_record(0)) + "\n{broken\n"
    fake = FakeS3({"k/bad.jsonl": good.encode()})
    store = make_store(tmp_path, fake)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/k/*.jsonl"))
    assert len(source.list_records()) == 1  # the probe only sees line 1
    with pytest.raises(InvalidTraceError, match=f"s3://{BUCKET}/k/bad.jsonl"):
        source.load_record(0)


# One of two objects changed after listing: skipped with skip_invalid_files, else a TraceReadError.
def test_unreadable_remote_object_is_skipped_or_fatal(tmp_path: Path) -> None:
    from inference_perf.datagen.replay.trace_source import TraceReadError

    fake = FakeS3({"k/a.jsonl": wire_bytes(compress=False), "k/b.jsonl": wire_bytes(compress=False)})
    store = make_store(tmp_path, fake)
    objs = store.resolve(f"s3://{BUCKET}/k/*.jsonl")
    fake.objects["k/a.jsonl"] += b"\n"  # changed after listing
    assert [r.source_id for r in remote_source(store, objs, skip_invalid=True).list_records()] == [objs[1].uri]
    with pytest.raises(TraceReadError, match="changed since it was listed"):
        remote_source(store, objs, skip_invalid=False)


# A local capture and a remote one in one source: source ids [local path, s3 URI], both load.
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


# An OTelTraceReplayDataGenerator over fake (installed as the default client) with its cache in
# tmp_path/cache.
def make_generator(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeS3,
    tmp_path: Path,
    trace_files: Optional[List[str]] = None,
    trace_directory: Optional[str] = None,
    remote_cache_max_gb: float = 10.0,
    duplicate_sessions_target: Optional[int] = None,
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
        duplicate_sessions_target=duplicate_sessions_target,
    )
    config = DataConfig(type="otel_trace_replay", otel_trace_replay=replay)
    return OTelTraceReplayDataGenerator(
        api_config=APIConfig(type=APIType.Chat, streaming=False), config=config, tokenizer=None, num_workers=1
    )


# The generator's remote store and LocalTraceSource, asserted present.
def remote_parts(gen: OTelTraceReplayDataGenerator) -> Tuple[RemoteTraceStore, LocalTraceSource]:
    assert gen._remote_store is not None and isinstance(gen._trace_source, LocalTraceSource)
    return gen._remote_store, gen._trace_source


# n wire captures m/run/task{i}-calls.jsonl.zst, two calls each, session header s{i}.
def corpus(n: int) -> FakeS3:
    return FakeS3({f"m/run/task{i}-calls.jsonl.zst": wire_bytes(2, session=f"s{i}") for i in range(n)})


# Three remote captures become three buildable sessions whose source ids are their URIs.
def test_generator_replays_remote_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gen = make_generator(monkeypatch, corpus(3), tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*-calls.jsonl.zst"])
    assert gen.get_session_count() == 3
    for i in range(3):
        assert gen.is_session_buildable(i)
    assert {s.source_id for s in gen.sessions if s is not None} == {
        f"s3://{BUCKET}/m/run/task{i}-calls.jsonl.zst" for i in range(3)
    }


# Three objects that fit the cache: probe + pre-download is 2 GETs each; parent builds, a pickled
# worker's builds and a cycling replay add none, and nobody waits.
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


# Six objects, cache for 4.5: warm-up caches the first two in play order (half the cache);
# building all six waits at most once and disk stays within the cap plus one object.
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


# A pickled worker building an already-cached session makes no GET at all.
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


# trace_directory s3://bkt/d/ replays d/a.jsonl only: not d/sub/b.jsonl, not d/notes.txt.
def test_remote_trace_directory_lists_direct_children_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": wire_bytes(compress=False), "d/sub/b.jsonl": wire_bytes(compress=False), "d/notes.txt": b"x"})
    gen = make_generator(monkeypatch, fake, tmp_path, trace_directory=f"s3://{BUCKET}/d/")
    assert gen.get_session_count() == 1


# d/* matching d/notes.txt raises the local "must be a .json or .jsonl file" error naming its URI.
def test_remote_match_without_trace_suffix_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": wire_bytes(compress=False), "d/notes.txt": b"x"})
    with pytest.raises(ValueError, match=r"must be a \.json or \.jsonl file.*s3://bkt/d/notes\.txt"):
        make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/d/*"])


# d/a.jsonl and d/a.jsonl.zst both listed: one session, not two.
def test_remote_compressed_duplicate_is_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({"d/a.jsonl": wire_bytes(compress=False), "d/a.jsonl.zst": wire_bytes()})
    gen = make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/d/*.jsonl*"])
    assert gen.get_session_count() == 1


# A remote pattern matching nothing raises "No trace files resolved", as locally.
def test_remote_pattern_with_no_matches_fails_like_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No trace files resolved"):
        make_generator(monkeypatch, corpus(1), tmp_path, trace_files=[f"s3://{BUCKET}/nope/*.jsonl.zst"])


# s3:///k/*.jsonl and s3:// fail config validation ("missing a bucket"); remote_cache_max_gb
# defaults to 10.
def test_config_rejects_s3_uri_without_bucket() -> None:
    from pydantic import ValidationError

    from inference_perf.config import OTelTraceReplayConfig

    with pytest.raises(ValidationError, match="missing a bucket"):
        OTelTraceReplayConfig(trace_files=["s3:///k/*.jsonl"])
    with pytest.raises(ValidationError, match="missing a bucket"):
        OTelTraceReplayConfig(trace_directory="s3://")
    assert OTelTraceReplayConfig(trace_files=["s3://b/k/*.jsonl"]).remote_cache_max_gb == 10.0


# --------------------------------------------------------------------------------------
# Failure handling and concurrency
# --------------------------------------------------------------------------------------


# materialize(o) on a thread; fails the test if it hangs past timeout, else returns its result or error.
def _materialize_in_thread(store: RemoteTraceStore, o: RemoteObject, timeout: float = 5.0) -> object:
    import threading

    box: Dict[str, object] = {}

    def run() -> None:
        try:
            box["result"] = store.materialize(o)
        except Exception as e:
            box["result"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "materialize hung"
    return box["result"]


# Cache dir path is a regular file: prefetch returns, and the read fails with RemoteReadError
# instead of hanging.
def test_prefetch_into_an_unwritable_cache_does_not_hang(tmp_path: Path) -> None:
    import threading

    fake = FakeS3({"a.jsonl": b"a"})
    blocker = tmp_path / "cache"
    blocker.write_text("a file where the cache directory should be")
    store = RemoteTraceStore(blocker, 10**6, client_factory=lambda: fake)
    done = threading.Event()

    def start_prefetch() -> None:
        store.prefetch([obj(fake, "a.jsonl")])
        done.set()

    threading.Thread(target=start_prefetch, daemon=True).start()
    assert done.wait(5), "prefetch hung"
    assert isinstance(_materialize_in_thread(store, obj(fake, "a.jsonl")), RemoteReadError)


# The first GET fails with InternalError: the read that needs the object still gets its bytes.
def test_a_failed_prefetch_is_retried_by_the_load_that_needs_it(tmp_path: Path) -> None:
    class FlakyOnce(FakeS3):
        failed = False

        def get_object(
            self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None
        ) -> GetObjectOutput:
            if not self.failed:
                self.failed = True
                raise ClientError({"Error": {"Code": "InternalError", "Message": "reset"}}, "GetObject")
            return super().get_object(Bucket, Key, IfMatch, Range)

    fake = FlakyOnce({"a.jsonl": b"a"})
    store = make_store(tmp_path, fake)
    store.prefetch([obj(fake, "a.jsonl")])
    result = _materialize_in_thread(store, obj(fake, "a.jsonl"))
    assert isinstance(result, Path) and result.read_bytes() == b"a"


# A process queues 32 slow (0.5 s) downloads and returns: it exits within the 20 s timeout
# instead of finishing them.
def test_exit_does_not_wait_for_queued_prefetches(tmp_path: Path) -> None:
    import subprocess

    script = f"""
import sys, time
sys.path.insert(0, {str(Path(__file__).parent)!r})
from pathlib import Path
from test_remote_trace_store import FakeS3, RemoteObject, RemoteTraceStore, GetObjectOutput

class Slow(FakeS3):
    def get_object(self, Bucket, Key, IfMatch=None, Range=None):
        time.sleep(0.5)
        return super().get_object(Bucket, Key, IfMatch, Range)

fake = Slow({{f"k{{i}}.jsonl": b"x" for i in range(32)}})
store = RemoteTraceStore(Path({str(tmp_path / "cache")!r}), 10**6, client_factory=lambda: fake)
store.prefetch([RemoteObject("bkt", k, fake.etag(k), 1) for k in fake.objects])
print("main done", flush=True)
"""
    # Non-daemon download threads would keep the interpreter alive past the timeout.
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=20)


# Old set a cached, newer b; pre-downloading a + c (1000 B, the whole cap) keeps every a and c.
def test_cached_objects_of_the_current_set_are_not_evicted_by_its_own_download(tmp_path: Path) -> None:
    objects = {f"a{i}.jsonl": b"a" * 150 for i in range(4)}
    objects.update({f"b{i}.jsonl": b"b" * 200 for i in range(2)})
    objects.update({f"c{i}.jsonl": b"c" * 100 for i in range(4)})
    fake = FakeS3(objects)
    store = make_store(tmp_path, fake, max_bytes=1000)
    a = [obj(fake, f"a{i}.jsonl") for i in range(4)]
    b = [obj(fake, f"b{i}.jsonl") for i in range(2)]
    c = [obj(fake, f"c{i}.jsonl") for i in range(4)]
    store.download_all(a)  # an earlier run
    time_after_a = max(store.cache_path(o).stat().st_mtime for o in a)
    for o in b:
        store.download_all([o])
        os.utime(store.cache_path(o), (time_after_a + 10, time_after_a + 10))
    store.download_all(a + c)  # this run: 1000 bytes, fits the cap
    assert all(store.cache_path(o).exists() for o in a + c)


# Listing refused with InvalidAccessKeyId gets the AWS_ACCESS_KEY_ID hint too.
def test_wrong_credentials_get_the_same_hint(tmp_path: Path) -> None:
    class BadKey(FakeS3):
        def paginate(self, Bucket: str, Prefix: str = "", Delimiter: str = "") -> Iterator[ListPage]:
            raise ClientError({"Error": {"Code": "InvalidAccessKeyId", "Message": "no such key"}}, "ListObjectsV2")

    with pytest.raises(ValueError, match=r"InvalidAccessKeyId.*AWS_ACCESS_KEY_ID"):
        make_store(tmp_path, BadKey({})).resolve(f"s3://{BUCKET}/*.jsonl")


# default_client builds an s3 client whose pool has room for the 16 + 8 download threads.
def test_default_client_pool_covers_the_download_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    import inference_perf.datagen.replay.remote_trace_store as remote_module

    seen: Dict[str, object] = {}
    monkeypatch.setattr(boto3, "client", lambda service, **kwargs: seen.update(service=service, **kwargs))
    remote_module.default_client()
    from botocore.config import Config

    config = seen["config"]
    assert seen["service"] == "s3" and isinstance(config, Config)
    assert config.max_pool_connections >= 16 + 8


# Eight OTel .json objects at 0.2 s each index in under 1 s (one at a time would take 1.6 s).
def test_remote_otel_objects_download_concurrently_at_indexing(tmp_path: Path) -> None:
    import time

    class Slow(FakeS3):
        def get_object(
            self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None
        ) -> GetObjectOutput:
            time.sleep(0.2)
            return super().get_object(Bucket, Key, IfMatch, Range)

    fake = Slow({f"d/t{i}.json": json.dumps(otel_doc(f"t{i}")).encode() for i in range(8)})
    store = make_store(tmp_path, fake)
    objs = store.resolve(f"s3://{BUCKET}/d/*.json")
    started = time.monotonic()
    assert len(remote_source(store, objs).list_records()) == 8
    assert time.monotonic() - started < 1.0  # 8 x 0.2s one at a time would be 1.6s


# Everything pre-downloaded: building sessions never calls prefetch.
def test_no_download_ahead_once_everything_was_pre_downloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gen = make_generator(monkeypatch, corpus(3), tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"])
    store, _ = remote_parts(gen)
    calls: List[int] = []
    monkeypatch.setattr(store, "prefetch", lambda objs: calls.append(len(objs)))
    for i in range(3):
        gen.is_session_buildable(i)
    assert calls == []


# A second run over a warm cache logs "All 3 remote traces are already cached" and no download.
def test_warm_run_reports_nothing_to_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    fake = corpus(3)
    make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"])
    caplog.clear()
    with caplog.at_level(logging.INFO):
        make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"])
    assert "All 3 remote traces are already cached" in caplog.text
    assert "Downloading" not in caplog.text


# --------------------------------------------------------------------------------------
# Cache accounting at scale
# --------------------------------------------------------------------------------------


# Total bytes of all files under root.
def disk_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# Pre-downloading 1000 small objects takes under 5 s (rescanning the cache per download took ~17 s).
def test_pre_downloading_many_objects_does_not_rescan_the_cache_each_time(tmp_path: Path) -> None:
    import time

    fake = FakeS3({f"k{i:04}.jsonl": b"x" * 8 for i in range(1000)})
    store = make_store(tmp_path, fake)
    started = time.monotonic()
    store.download_all([obj(fake, k) for k in fake.objects])
    assert time.monotonic() - started < 5.0  # a rescan per download made this ~17s


# Cap 1000 B, an old pinned-then-released 1000 B set, then a new 1000 B pinned set: the new
# set is complete and disk holds at most 1000 B.
def test_pinned_files_count_against_the_cap(tmp_path: Path) -> None:
    fake = FakeS3({**{f"a{i}.jsonl": b"a" * 100 for i in range(10)}, **{f"b{i}.jsonl": b"b" * 100 for i in range(10)}})
    store = make_store(tmp_path, fake, max_bytes=1000)
    a = [obj(fake, f"a{i}.jsonl") for i in range(10)]
    b = [obj(fake, f"b{i}.jsonl") for i in range(10)]
    store.pin(a)
    store.download_all(a)  # an earlier run's corpus, 1000 bytes
    store.unpin(a)
    store.pin(b)
    store.download_all(b)  # this run's corpus, also 1000 bytes
    assert all(store.cache_path(o).exists() for o in b)
    assert disk_bytes(tmp_path / "cache") <= 1000


# Ten ~1 KB OTel objects with a 2500 B cap: all ten index and disk stays within the cap plus one.
def test_indexing_remote_otel_respects_the_cap(tmp_path: Path) -> None:
    fake = FakeS3({f"d/t{i}.json": json.dumps({**otel_doc(f"t{i}"), "pad": "x" * 1000}).encode() for i in range(10)})
    store = make_store(tmp_path, fake, max_bytes=2500)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.json"))
    assert len(source.list_records()) == 10
    assert disk_bytes(tmp_path / "cache") <= 2500 + 1100
    assert source.load_record(9)["trace_id"] == "t9"


# A .part file from 1970 inside a cache entry's dir is removed by the next scan.
def test_a_stale_part_file_is_cleaned_up(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a"})
    stale = tmp_path / "cache" / f"{'0' * 24}-1" / ".1.2.part"  # inside a cache entry's dir
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"partial")
    os.utime(stale, (0, 0))
    make_store(tmp_path, fake).materialize(obj(fake, "a.jsonl"))
    assert not stale.exists()


# The cached file eviction picks sits in a read-only dir: the next download still succeeds.
def test_an_undeletable_cached_file_does_not_fail_a_download(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a" * 100, "b.jsonl": b"b" * 100})
    store = make_store(tmp_path, fake, max_bytes=150)
    locked = store.materialize(obj(fake, "a.jsonl"))
    locked.parent.chmod(0o500)
    try:
        assert store.materialize(obj(fake, "b.jsonl")).read_bytes() == b"b" * 100
    finally:
        locked.parent.chmod(0o700)


# --------------------------------------------------------------------------------------
# Reads: retries, ranged probes
# --------------------------------------------------------------------------------------


# A body whose every read fails with a connection reset.
class _ResetBody(FakeBody):
    def read(self, amt: Optional[int] = None) -> bytes:
        raise ConnectionResetError("connection reset by peer")


# The first download is reset mid-read: it is retried and succeeds on the second GET.
def test_a_transient_read_failure_is_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import inference_perf.datagen.replay.remote_trace_store as remote_module

    monkeypatch.setattr(remote_module, "_RETRY_DELAYS", (0.0, 0.0))

    class ResetOnce(FakeS3):
        reset = False

        def get_object(
            self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None
        ) -> GetObjectOutput:
            out = super().get_object(Bucket, Key, IfMatch, Range)
            if not self.reset:
                self.reset = True
                return {"Body": _ResetBody(b"", self)}
            return out

    fake = ResetOnce({"a.jsonl": b"a"})
    assert make_store(tmp_path, fake).materialize(obj(fake, "a.jsonl")).read_bytes() == b"a"
    assert len(fake.gets) == 2


# NoSuchKey is permanent: one GET, then RemoteReadError.
def test_a_missing_object_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import inference_perf.datagen.replay.remote_trace_store as remote_module

    monkeypatch.setattr(remote_module, "_RETRY_DELAYS", (0.0, 0.0))
    fake = FakeS3({})
    with pytest.raises(RemoteReadError):
        make_store(tmp_path, fake).materialize(RemoteObject(BUCKET, "gone.jsonl", '"e"', 1))
    assert fake.gets == ["gone.jsonl"]


# The probe asks for bytes 0-262143 and reads that range to its end, so the connection is reused.
def test_probe_reads_a_bounded_range_to_its_end(tmp_path: Path) -> None:
    data = wire_bytes(400)
    fake = FakeS3({"k.jsonl.zst": data})
    assert make_store(tmp_path, fake).first_record(obj(fake, "k.jsonl.zst")) == wire_record(0)
    assert fake.ranges == ["bytes=0-262143"]
    assert fake.bytes_served == min(len(data), 262144)  # read to the end, so the connection is reused


# A first record of ~300 KB, longer than the first 256 KiB range, is still returned whole.
def test_probe_finds_a_first_line_longer_than_the_first_range(tmp_path: Path) -> None:
    first = {**wire_record(0), "request": os.urandom(150_000).hex()}
    data = zstd.compress((json.dumps(first) + "\n" + json.dumps(wire_record(1)) + "\n").encode())
    fake = FakeS3({"k.jsonl.zst": data})
    assert make_store(tmp_path, fake).first_record(obj(fake, "k.jsonl.zst")) == first


# A .zst cut in half is a RemoteReadError naming its URI and "truncated".
def test_probe_of_a_truncated_zst_is_a_read_error(tmp_path: Path) -> None:
    data = wire_bytes(2)
    fake = FakeS3({"k.jsonl.zst": data[: len(data) // 2]})
    with pytest.raises(RemoteReadError, match=r"s3://bkt/k\.jsonl\.zst.*truncated"):
        make_store(tmp_path, fake).first_record(obj(fake, "k.jsonl.zst"))


# Every probe refused with AccessDenied, even with skip_invalid_files: a ValueError with the
# AWS_ACCESS_KEY_ID hint.
def test_every_probe_denied_names_the_credentials(tmp_path: Path) -> None:
    class Denied(FakeS3):
        def get_object(
            self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None
        ) -> GetObjectOutput:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject")

    fake = Denied({"k/a.jsonl": b"x", "k/b.jsonl": b"y"})
    store = make_store(tmp_path, fake)
    with pytest.raises(ValueError, match=r"AccessDenied.*AWS_ACCESS_KEY_ID"):
        remote_source(store, store.resolve(f"s3://{BUCKET}/k/*.jsonl"), skip_invalid=True)


# A cached .jsonl deleted by another process loads again; a read that loses its file between
# hit and open is retried once and returns t0.
def test_vanished_cache_hit_is_fetched_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from inference_perf.datagen.replay.trace_source import TraceReadError

    body = (json.dumps(otel_doc("t0")) + "\n" + json.dumps(otel_doc("t1")) + "\n").encode()
    fake = FakeS3({"d/traces.jsonl": body})
    store = make_store(tmp_path, fake)
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.jsonl"))
    evicted = source.records[1].remote
    assert evicted is not None
    store.cache_path(evicted).unlink()  # evicted by another process
    assert source.load_record(1)["trace_id"] == "t1"

    real = source._read_raw
    calls: List[int] = []

    def vanish_once(record: LocalTraceRecord) -> Dict[str, object]:
        calls.append(1)
        if len(calls) == 1:
            raise TraceReadError("gone") from FileNotFoundError()
        return real(record)

    monkeypatch.setattr(source, "_read_raw", vanish_once)
    assert source.load_record(0)["trace_id"] == "t0" and len(calls) == 2


# --------------------------------------------------------------------------------------
# Generator: download-ahead, directories, cache dir
# --------------------------------------------------------------------------------------


# Build sessions 0..count-1 like the parent dispatch loop, letting download-ahead finish each time.
def build_in_order(gen: OTelTraceReplayDataGenerator, count: int) -> RemoteTraceStore:
    store, _ = remote_parts(gen)
    for i in range(count):
        gen.is_session_buildable(i)
        wait_until_cached(store, [])
    return store


# A cache budget, in GB, of `objects` times the size of one object in fake.
def small_cap(fake: FakeS3, objects: float) -> float:
    return objects * len(next(iter(fake.objects.values()))) / 1e9


# Six objects, cache for 4.5, three passes over the corpus (18 sessions): no session waits.
def test_download_ahead_follows_corpus_cycling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = corpus(6)
    gen = make_generator(
        monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_max_gb=small_cap(fake, 4.5)
    )
    assert build_in_order(gen, 18).sync_downloads == 0  # three passes over the corpus


# Four objects padded to 12 sessions, cache for 2.5: all 12 sessions, none waits.
def test_download_ahead_follows_duplicated_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = corpus(4)
    gen = make_generator(
        monkeypatch,
        fake,
        tmp_path,
        trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"],
        remote_cache_max_gb=small_cap(fake, 2.5),
        duplicate_sessions_target=12,
    )
    assert gen.get_session_count() == 12
    assert build_in_order(gen, 12).sync_downloads == 0


# 40 objects, window of ~10: after warm-up each dispatch queues at most 2 new objects.
def test_each_dispatch_queues_only_the_new_end_of_the_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = corpus(40)
    gen = make_generator(
        monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_max_gb=small_cap(fake, 20.5)
    )
    store, _ = remote_parts(gen)
    sizes: List[int] = []
    real = store.prefetch

    def spy(objs: Sequence[RemoteObject]) -> None:
        sizes.append(len(objs))
        real(objs)

    monkeypatch.setattr(store, "prefetch", spy)
    for i in range(10):
        gen.is_session_buildable(i)
    assert sizes and max(sizes) <= 2


# One capture larger than half the cache is still admitted to the window: five sessions, no waits.
def test_an_object_over_half_the_cap_is_still_downloaded_ahead(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    objects = {f"m/run/task{i}-calls.jsonl.zst": wire_bytes(2, session=f"s{i}") for i in range(4)}
    objects["m/run/task9-calls.jsonl.zst"] = wire_bytes(60, session="big")
    fake = FakeS3(objects)
    big = len(objects["m/run/task9-calls.jsonl.zst"])
    gen = make_generator(
        monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_max_gb=1.2 * big / 1e9
    )
    assert build_in_order(gen, 5).sync_downloads == 0


# Downloads made while indexing OTel objects at startup are not counted as waits.
def test_setup_downloads_are_not_reported_as_waits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({f"d/t{i}.json": json.dumps({**otel_doc(f"t{i}"), "pad": "x" * 1000}).encode() for i in range(6)})
    gen = make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/d/*.json"], remote_cache_max_gb=2500 / 1e9)
    assert remote_parts(gen)[0].sync_downloads == 0


# remote_cache_max_gb 0: three sessions build without a single "Waited" warning.
def test_cap_zero_does_not_warn_on_every_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gen = make_generator(
        monkeypatch, corpus(3), tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_max_gb=0
    )
    caplog.clear()
    for i in range(3):
        assert gen.is_session_buildable(i)
    assert "Waited" not in caplog.text


# trace_directory s3://bkt/run[1]/: brackets are literal, .b-calls.jsonl counts, sub/ does not:
# two sessions from one delimited listing of run[1]/.
def test_remote_directory_is_literal_and_includes_dotfiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3(
        {
            "run[1]/a-calls.jsonl": wire_bytes(compress=False),
            "run[1]/.b-calls.jsonl": wire_bytes(compress=False),
            "run[1]/sub/c-calls.jsonl": wire_bytes(compress=False),
        }
    )
    gen = make_generator(monkeypatch, fake, tmp_path, trace_directory=f"s3://{BUCKET}/run[1]/")
    assert gen.get_session_count() == 2
    assert fake.listed == [("run[1]/", "/")]


# runs-2026*/*.jsonl lists with prefix runs-2026 and matches runs-2026a/x and runs-2026b/y.
def test_listing_uses_the_literal_part_of_the_first_wildcard_segment(tmp_path: Path) -> None:
    fake = FakeS3({"runs-2026a/x.jsonl": b"", "runs-2026b/y.jsonl": b"", "other/z.jsonl": b""})
    found = make_store(tmp_path, fake).resolve(f"s3://{BUCKET}/runs-2026*/*.jsonl")
    assert [o.key for o in found] == ["runs-2026a/x.jsonl", "runs-2026b/y.jsonl"]
    assert fake.listed == [("runs-2026", "")]


# k//x.jsonl and k/x.jsonl are different objects: two sessions.
def test_keys_that_normalize_alike_are_both_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeS3({"k//x.jsonl": wire_bytes(compress=False), "k/x.jsonl": wire_bytes(compress=False)})
    gen = make_generator(monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/k/x.jsonl", f"s3://{BUCKET}/k//x.jsonl"])
    assert gen.get_session_count() == 2


# An unsupported wire path (/v1/embeddings) is reported with the object's s3:// URI.
def test_errors_name_the_object_uri(tmp_path: Path) -> None:
    fake = FakeS3({"k/odd.jsonl": json.dumps({"path": "/v1/embeddings", "request": "{}"}).encode() + b"\n"})
    store = make_store(tmp_path, fake)
    with pytest.raises(ValueError, match=r"s3://bkt/k/odd\.jsonl"):
        remote_source(store, store.resolve(f"s3://{BUCKET}/k/*.jsonl"))


# remote_cache_dir under a regular file fails generator construction with a remote_cache_dir error.
def test_unusable_cache_dir_fails_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import inference_perf.datagen.replay.remote_trace_store as remote_module
    from inference_perf.config import APIConfig, APIType, DataConfig, OTelTraceReplayConfig

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    _SHARED["fake"] = corpus(1)
    monkeypatch.setattr(remote_module, "default_client", _shared_client)
    replay = OTelTraceReplayConfig(trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_dir=str(blocker / "cache"))
    with pytest.raises(ValueError, match="remote_cache_dir"):
        OTelTraceReplayDataGenerator(
            api_config=APIConfig(type=APIType.Chat, streaming=False),
            config=DataConfig(type="otel_trace_replay", otel_trace_replay=replay),
            tokenizer=None,
        )


# default_client sets a connect timeout of at most 15 s.
def test_default_client_bounds_connect_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3
    from botocore.config import Config

    import inference_perf.datagen.replay.remote_trace_store as remote_module

    seen: Dict[str, object] = {}
    monkeypatch.setattr(boto3, "client", lambda service, **kwargs: seen.update(kwargs))
    remote_module.default_client()
    config = seen["config"]
    assert isinstance(config, Config) and config.connect_timeout <= 15


# Resolving d/*.jsonl logs "Listing s3://bkt/d/".
def test_listing_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO):
        make_store(tmp_path, FakeS3({"d/a.jsonl": b""})).resolve(f"s3://{BUCKET}/d/*.jsonl")
    assert f"Listing s3://{BUCKET}/d/" in caplog.text


# Download-ahead run synchronously (its evictions land first), cache for 1.2x the big capture:
# the session being built keeps its file, so five sessions build with no wait.
def test_the_session_being_built_keeps_its_file_while_the_window_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = {f"m/run/task{i}-calls.jsonl.zst": wire_bytes(2, session=f"s{i}") for i in range(4)}
    objects["m/run/task9-calls.jsonl.zst"] = wire_bytes(60, session="big")
    fake = FakeS3(objects)
    big = len(objects["m/run/task9-calls.jsonl.zst"])
    gen = make_generator(
        monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], remote_cache_max_gb=1.2 * big / 1e9
    )
    store, _ = remote_parts(gen)
    # Download ahead synchronously, so its evictions land before the session is read: the
    # worst case of the race between background downloads and the build that queued them.
    monkeypatch.setattr(store, "prefetch", store.download_all)
    for i in range(5):
        assert gen.is_session_buildable(i)
    assert store.sync_downloads == 0


# --------------------------------------------------------------------------------------
# Shared cache directory safety
# --------------------------------------------------------------------------------------


# A cache dir that also holds a user's own file (my-project/results.csv, 500 B) and a
# 150 B budget: downloads evict only the store's own entries, never results.csv.
def test_eviction_never_deletes_files_the_store_did_not_create(tmp_path: Path) -> None:
    own = tmp_path / "cache" / "my-project" / "results.csv"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"r" * 500)
    fake = FakeS3({"a.jsonl": b"a" * 100, "b.jsonl": b"b" * 100})
    store = make_store(tmp_path, fake, max_bytes=150)
    store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "b.jsonl"))
    assert own.read_bytes() == b"r" * 500


# The default cache dir under the system temp dir is created private (mode 0700).
def test_private_dir_is_created_owner_only(tmp_path: Path) -> None:
    from inference_perf.datagen.replay.remote_trace_store import prepare_private_dir

    target = tmp_path / "inference-perf-me"
    prepare_private_dir(target)
    assert target.stat().st_mode & 0o777 == 0o700


# A default cache dir that is a symlink (planted by another user) is refused.
def test_private_dir_refuses_a_symlink(tmp_path: Path) -> None:
    from inference_perf.datagen.replay.remote_trace_store import prepare_private_dir

    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "inference-perf-me").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        prepare_private_dir(tmp_path / "inference-perf-me")


# A default cache dir owned by another uid is refused.
def test_private_dir_refuses_another_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import inference_perf.datagen.replay.remote_trace_store as remote_module

    target = tmp_path / "inference-perf-me"
    target.mkdir(mode=0o700)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(target).st_uid + 1)
    with pytest.raises(ValueError, match="owned by"):
        remote_module.prepare_private_dir(target)


# A default cache dir that other users can write to (mode 0777) is refused.
def test_private_dir_refuses_one_others_can_write(tmp_path: Path) -> None:
    from inference_perf.datagen.replay.remote_trace_store import prepare_private_dir

    target = tmp_path / "inference-perf-me"
    target.mkdir()
    target.chmod(0o777)
    with pytest.raises(ValueError, match="writable by other users"):
        prepare_private_dir(target)


# With no remote_cache_dir, the generator's default cache lives in a private per-user dir.
def test_generator_default_cache_dir_is_private(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tempfile

    import inference_perf.datagen.replay.remote_trace_store as remote_module
    from inference_perf.config import APIConfig, APIType, DataConfig, OTelTraceReplayConfig

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _SHARED["fake"] = corpus(1)
    monkeypatch.setattr(remote_module, "default_client", _shared_client)
    replay = OTelTraceReplayConfig(trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"])
    gen = OTelTraceReplayDataGenerator(
        api_config=APIConfig(type=APIType.Chat, streaming=False),
        config=DataConfig(type="otel_trace_replay", otel_trace_replay=replay),
        tokenizer=None,
    )
    store, _ = remote_parts(gen)
    base = store.cache_dir.parent
    assert base.parent == tmp_path and base.stat().st_mode & 0o777 == 0o700


# A pickled (worker) copy that misses with a 150 B budget leaves the parent's pinned files alone.
def test_a_worker_copy_never_evicts(tmp_path: Path) -> None:
    fake = FakeS3({"a.jsonl": b"a" * 100, "b.jsonl": b"b" * 100, "c.jsonl": b"c" * 100})
    _SHARED["fake"] = fake
    parent = RemoteTraceStore(tmp_path / "cache", 150, client_factory=_shared_client)
    parent.pin([obj(fake, "a.jsonl")])
    parent.materialize(obj(fake, "a.jsonl"))
    worker = pickle.loads(pickle.dumps(parent))
    worker.materialize(obj(fake, "c.jsonl"))
    assert parent.cache_path(obj(fake, "a.jsonl")).exists()


# A 100 B download into a full 300 B cache makes room before it starts writing, so disk
# never holds the old 300 B plus the partial file.
def test_room_is_made_before_a_download_writes(tmp_path: Path) -> None:
    import threading

    release = threading.Event()
    started = threading.Event()

    class Blocking(FakeBody):
        def read(self, amt: Optional[int] = None) -> bytes:
            started.set()
            release.wait(5)
            return super().read(amt)

    class SlowNew(FakeS3):
        def get_object(
            self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None
        ) -> GetObjectOutput:
            out = super().get_object(Bucket, Key, IfMatch, Range)
            return {"Body": Blocking(self.objects[Key], self)} if Key == "new.jsonl" else out

    fake = SlowNew({**{f"old{i}.jsonl": b"o" * 100 for i in range(3)}, "new.jsonl": b"n" * 100})
    store = make_store(tmp_path, fake, max_bytes=300)
    for i in range(3):
        store.materialize(obj(fake, f"old{i}.jsonl"))
    worker = threading.Thread(target=store.materialize, args=(obj(fake, "new.jsonl"),), daemon=True)
    worker.start()
    assert started.wait(5)
    complete = sum(p.stat().st_size for p in (tmp_path / "cache").rglob("*.jsonl"))
    release.set()
    worker.join(5)
    assert complete <= 200


# Ten 1 KB OTel objects with room for 9.5: indexing downloads each object exactly once.
def test_otel_set_just_over_the_cap_is_downloaded_once(tmp_path: Path) -> None:
    docs = {f"d/t{i}.json": json.dumps({**otel_doc(f"t{i}"), "pad": "x" * 1000}).encode() for i in range(10)}
    fake = FakeS3(docs)
    size = len(next(iter(docs.values())))
    store = make_store(tmp_path, fake, max_bytes=int(9.5 * size))
    source = remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.json"))
    assert len(source.list_records()) == 10
    assert len(fake.gets) == 10


# OTel .json objects all refused (AccessDenied): fail with the credentials hint, and try
# each object only once.
def test_otel_objects_all_denied_name_the_credentials(tmp_path: Path) -> None:
    class Denied(FakeS3):
        def get_object(
            self, Bucket: str, Key: str, IfMatch: Optional[str] = None, Range: Optional[str] = None
        ) -> GetObjectOutput:
            self.gets.append(Key)
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject")

    fake = Denied({"d/a.json": b"{}", "d/b.json": b"{}"})
    store = make_store(tmp_path, fake)
    with pytest.raises(ValueError, match=r"AccessDenied.*AWS_ACCESS_KEY_ID"):
        remote_source(store, store.resolve(f"s3://{BUCKET}/d/*.json"), skip_invalid=True)
    assert sorted(fake.gets) == ["d/a.json", "d/b.json"]


# A zero-byte object probes as empty without a request (a ranged GET of it returns 416).
def test_an_empty_object_probes_as_empty(tmp_path: Path) -> None:
    fake = FakeS3({"k/empty.jsonl": b""})
    assert make_store(tmp_path, fake).first_record(obj(fake, "k/empty.jsonl")) is None
    assert fake.gets == []


# Another process removes the entry dir between mkdir and opening the temp file: the
# download is retried, not reported as an unwritable cache.
def test_entry_dir_removed_before_the_temp_file_opens_is_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import inference_perf.datagen.replay.remote_trace_store as remote_module

    monkeypatch.setattr(remote_module, "_RETRY_DELAYS", (0.0, 0.0))
    real_mkdir = Path.mkdir
    calls: List[int] = []

    # Create the entry dir, then remove it once, as another process's eviction would.
    def mkdir_then_vanish(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        real_mkdir(self, mode, parents, exist_ok)
        if self.parent == tmp_path / "cache" and not calls:
            calls.append(1)
            self.rmdir()

    monkeypatch.setattr(Path, "mkdir", mkdir_then_vanish)
    fake = FakeS3({"a.jsonl": b"a"})
    assert make_store(tmp_path, fake).materialize(obj(fake, "a.jsonl")).read_bytes() == b"a"
    assert len(fake.gets) == 2


# A wire capture stored as .json has a session id from its URI, the same for any cache dir.
def test_remote_json_wire_capture_id_comes_from_the_uri(tmp_path: Path) -> None:
    fake = FakeS3({"k/calls.json": json.dumps(wire_record(0)).encode()})
    uri = f"s3://{BUCKET}/k/calls.json"
    for sub in ("one", "two"):
        store = RemoteTraceStore(tmp_path / sub, 10**9, client_factory=lambda: fake)
        [record] = remote_source(store, store.resolve(uri)).list_records()
        assert record.session_id_suffix == hashlib.sha256(uri.encode()).hexdigest()[:32]


# Each wait warning carries the totals so far, so a run's stalls can be read off its last one.
def test_wait_warnings_carry_running_totals(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeS3({"a.jsonl": b"a", "b.jsonl": b"b"})
    store = make_store(tmp_path, fake)
    store.materialize(obj(fake, "a.jsonl"))
    store.materialize(obj(fake, "b.jsonl"))
    assert "2 waits" in caplog.text


# check_not_all_denied raises only when every result is a refusal with one credential code.
@pytest.mark.parametrize(
    "results,raises",
    [
        ([], False),
        ([RemoteReadError("x", code="AccessDenied")], True),
        ([RemoteReadError("x", code="AccessDenied"), RemoteReadError("y", code="InvalidAccessKeyId")], False),
        ([RemoteReadError("x", code="AccessDenied"), {"path": "/v1/messages"}], False),
        ([RemoteReadError("x", code="NoSuchKey")], False),
        ([ValueError("bad json")], False),
    ],
)
def test_check_not_all_denied_conditions(results: List[Union[Dict[str, object], None, Exception]], raises: bool) -> None:
    from inference_perf.datagen.replay.remote_trace_store import check_not_all_denied

    if raises:
        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            check_not_all_denied(results)
    else:
        check_not_all_denied(results)


# 4 traces padded to 6 sessions, then cycled to index 12: every index builds, none crashes.
def test_cycling_a_padded_corpus_builds_every_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gen = make_generator(
        monkeypatch, corpus(4), tmp_path, trace_files=[f"s3://{BUCKET}/m/*/*.jsonl.zst"], duplicate_sessions_target=6
    )
    for i in range(13):
        assert gen.is_session_buildable(i)


# One multi-row .jsonl shared by six sessions plus two single files, cache for 2.5 files:
# the shared file is charged once, so the window holds every session up to a new file
# that would not fit, not just one session.
def test_window_charges_a_shared_file_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = "".join(json.dumps(otel_doc(f"r{i}")) + "\n" for i in range(6)).encode()
    size = len(rows)
    objects = {
        "d/many.jsonl": rows,
        "d/a.json": json.dumps({**otel_doc("a"), "pad": "x" * (size - 40)}).encode(),
        "d/b.json": json.dumps({**otel_doc("b"), "pad": "x" * (size - 40)}).encode(),
    }
    fake = FakeS3(objects)
    gen = make_generator(
        monkeypatch, fake, tmp_path, trace_files=[f"s3://{BUCKET}/d/*.json*"], remote_cache_max_gb=2.5 * size / 1e9
    )
    store, source = remote_parts(gen)
    budget = store.max_bytes / 2
    expected, total = 0, 0
    seen: set[str] = set()
    for index in range(gen.get_session_count()):
        record = gen._record_for_index(index)
        assert record is not None and record.remote is not None
        if record.remote.uri not in seen:
            if seen and total + record.remote.size > budget:
                break
            seen.add(record.remote.uri)
            total += record.remote.size
        expected += 1
    assert len(gen._ahead) == expected
    assert gen._ahead_bytes == total
