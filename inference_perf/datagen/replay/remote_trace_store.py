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

"""Replay traces stored in S3-compatible object storage.

``s3://bucket/pattern`` entries are listed with glob semantics, probed for their first
record, and downloaded into a bounded local cache on demand. Parsing stays with the local
trace source: this module only hands back local paths.

Credentials and the endpoint come from boto3's standard chain (``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY``, ``AWS_ENDPOINT_URL_S3``, or a profile), so nothing here names
a provider and nothing secret ever reaches the config.

The cache exists because one session's trace is read more than once: by the parent when
it dispatches the session, by the worker it is pinned to, and again on every corpus-cycling
replay. Cached files keep their compressed bytes and their original basename, so format
detection and fallback identities behave exactly as for a local file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from inference_perf.datagen.replay.wire_trace_converter import _zstd

logger = logging.getLogger(__name__)

S3_SCHEME = "s3://"
_PROBE_CHUNK = 64 * 1024
_DOWNLOAD_CHUNK = 1024 * 1024
_PROBE_THREADS = 16
_PREFETCH_THREADS = 8


class RemoteReadError(OSError):
    """An object could not be read. An OSError, so callers map it like any other read failure."""


def is_remote(entry: str) -> bool:
    return entry.startswith(S3_SCHEME)


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split ``s3://bucket/key`` into (bucket, key); the key may be empty."""
    if not is_remote(uri):
        raise ValueError(f"Not an s3:// URI: {uri}")
    bucket, _, key = uri[len(S3_SCHEME) :].partition("/")
    if not bucket:
        raise ValueError(f"S3 URI is missing a bucket: {uri}")
    return bucket, key


def has_wildcard(entry: str) -> bool:
    """Whether an entry is a pattern; the same rule resolve_trace_files applies to local paths."""
    return "*" in entry or "?" in entry


def _is_magic(segment: str) -> bool:
    return any(c in segment for c in "*?[")


def _match_segments(patterns: Sequence[str], segments: Sequence[str]) -> bool:
    if not patterns:
        return not segments
    head = patterns[0]
    if head == "**":
        # Zero or more whole segments, never descending into a hidden one (glob's rule). A
        # trailing ** matching nothing names a directory, and keys are files, so it needs one.
        for consumed in range(0 if len(patterns) > 1 else 1, len(segments) + 1):
            if consumed and segments[consumed - 1].startswith("."):
                return False
            if _match_segments(patterns[1:], segments[consumed:]):
                return True
        return False
    if not segments:
        return False
    segment = segments[0]
    if _is_magic(head):
        if segment.startswith(".") and not head.startswith("."):
            return False
        if not fnmatchcase(segment, head):
            return False
    elif segment != head:
        return False
    return _match_segments(patterns[1:], segments[1:])


def match_key(pattern: str, key: str) -> bool:
    """Match an object key against a pattern the way ``glob.glob(recursive=True)`` matches files."""
    return _match_segments(pattern.split("/"), key.split("/"))


@dataclass(frozen=True)
class RemoteObject:
    """One listed object. The ETag pins the version every later read must see."""

    bucket: str
    key: str
    etag: str
    size: int

    @property
    def uri(self) -> str:
        return f"{S3_SCHEME}{self.bucket}/{self.key}"

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]


def default_client() -> Any:
    import boto3

    return boto3.client("s3")


class RemoteTraceStore:
    """Lists, probes and caches remote trace objects.

    Shared by the parent and every worker through a pickled copy. The copy rebuilds its
    client lazily and never prefetches: only the parent knows the play order.
    """

    def __init__(self, cache_dir: Path, max_bytes: int, client_factory: Optional[Callable[[], Any]] = None) -> None:
        self.cache_dir = cache_dir
        self.max_bytes = max_bytes
        # None looks default_client up at first use, so a replacement installed later applies.
        self.client_factory = client_factory
        self.prefetch_enabled = True
        # Off while the run is still being prepared, when downloads are expected rather than stalls.
        self.report_waits = True
        self.sync_downloads = 0
        self.sync_wait_s = 0.0
        self._init_runtime()

    def _init_runtime(self) -> None:
        self._client: Any = None
        self._lock = threading.Lock()
        self._inflight: Dict[str, Future[Path]] = {}
        self._pool: Optional[ThreadPoolExecutor] = None

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        for name in ("_client", "_lock", "_inflight", "_pool"):
            state.pop(name, None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_runtime()
        self.prefetch_enabled = False
        self.sync_downloads = 0
        self.sync_wait_s = 0.0

    @property
    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                self._client = (self.client_factory or default_client)()
            return self._client

    # ----------------------------------------------------------------------------------
    # Listing
    # ----------------------------------------------------------------------------------

    def resolve(self, uri: str) -> List[RemoteObject]:
        """Return the objects an entry names, sorted by key.

        A pattern with no matches returns []; an exact key that does not exist raises,
        like a missing local trace file.
        """
        bucket, key = parse_s3_uri(uri)
        if not has_wildcard(key):
            found = [o for o in self._list(bucket, key) if o.key == key]
            if not found:
                raise ValueError(f"Trace file does not exist: {uri}")
            return found
        segments = key.split("/")
        literal: List[str] = []
        for segment in segments[:-1]:
            if _is_magic(segment):
                break
            literal.append(segment)
        prefix = "/".join(literal) + "/" if literal else ""
        return sorted((o for o in self._list(bucket, prefix) if match_key(key, o.key)), key=lambda o: o.key)

    def _list(self, bucket: str, prefix: str) -> List[RemoteObject]:
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError

        where = f"{S3_SCHEME}{bucket}/{prefix}"
        try:
            objects: List[RemoteObject] = []
            for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    if not item["Key"].endswith("/"):
                        objects.append(RemoteObject(bucket, item["Key"], item["ETag"], int(item["Size"])))
            return objects
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise ValueError(
                f"No credentials to list {where}: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "(plus AWS_ENDPOINT_URL_S3 for a store other than AWS S3), or select a profile with AWS_PROFILE"
            ) from e
        except ClientError as e:
            error = e.response.get("Error", {})
            raise ValueError(f"Cannot list {where}: {error.get('Code')} {error.get('Message')}") from e
        except BotoCoreError as e:
            raise ValueError(f"Cannot list {where}: {e}. Check AWS_ENDPOINT_URL_S3 or the profile's endpoint_url") from e

    # ----------------------------------------------------------------------------------
    # Reads
    # ----------------------------------------------------------------------------------

    def _open(self, obj: RemoteObject) -> Any:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return self.client.get_object(Bucket=obj.bucket, Key=obj.key, IfMatch=obj.etag)["Body"]
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code"))
            if code in ("PreconditionFailed", "412"):
                raise RemoteReadError(f"{obj.uri} changed since it was listed") from e
            raise RemoteReadError(f"{obj.uri}: {code} {e.response.get('Error', {}).get('Message')}") from e
        except BotoCoreError as e:
            raise RemoteReadError(f"{obj.uri}: {e}") from e

    def first_record(self, obj: RemoteObject) -> Optional[Dict[str, Any]]:
        """Return the first non-blank line of a line-delimited object as a dict, reading only that far."""
        body = self._open(obj)
        zstd = _zstd() if obj.name.endswith(".zst") else None
        decompressor = zstd.ZstdDecompressor() if zstd else None
        pending = b""
        try:
            while True:
                try:
                    raw = body.read(_PROBE_CHUNK)
                    data = raw
                    while decompressor is not None and data:
                        out = decompressor.decompress(data)
                        # A .zst may hold several frames; each needs a fresh decompressor.
                        data = decompressor.unused_data if decompressor.eof else b""
                        if decompressor.eof:
                            decompressor = zstd.ZstdDecompressor() if zstd else None
                        pending += out
                    if decompressor is None and zstd is None:
                        pending += raw
                except Exception as e:  # network error or corrupt .zst
                    raise RemoteReadError(f"{obj.uri}: {e}") from e
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.strip():
                        return _parse_first(line, obj)
                if not raw:
                    return _parse_first(pending, obj) if pending.strip() else None
        finally:
            body.close()

    def first_records(self, objs: Sequence[RemoteObject]) -> List[Union[Dict[str, Any], None, Exception]]:
        """first_record for each object, concurrently; a failure is returned in its slot."""

        def probe(o: RemoteObject) -> Union[Dict[str, Any], None, Exception]:
            try:
                return self.first_record(o)
            except Exception as e:
                return e

        with ThreadPoolExecutor(_PROBE_THREADS, thread_name_prefix="trace-probe") as pool:
            return list(pool.map(probe, objs))

    # ----------------------------------------------------------------------------------
    # Cache
    # ----------------------------------------------------------------------------------

    def cache_path(self, obj: RemoteObject) -> Path:
        version = re.sub(r"[^A-Za-z0-9]", "", obj.etag)[:40]
        return self.cache_dir / f"{hashlib.sha256(obj.uri.encode('utf-8')).hexdigest()[:24]}-{version}" / obj.name

    def materialize(self, obj: RemoteObject) -> Path:
        """Return a local path holding the object, downloading it if it isn't cached.

        A download here blocks the caller, so each one is counted and logged: during a run
        it means the trace was not prefetched in time.
        """
        with self._lock:
            pending = self._inflight.get(obj.uri)
        path = self.cache_path(obj)
        if pending is None and path.exists():
            try:
                os.utime(path)
                return path
            except FileNotFoundError:
                pass  # evicted by another process just now
        started = time.perf_counter()
        if pending is not None:
            if pending.done():
                return pending.result()
            pending.result()
            reason = "its prefetch had not finished"
        else:
            self._fetch(obj)
            reason = "it was not prefetched"
        waited = time.perf_counter() - started
        if self.report_waits:
            self.sync_downloads += 1
            self.sync_wait_s += waited
            logger.warning(f"Waited {waited:.2f}s for {obj.uri} ({obj.size} bytes): {reason}")
        return path

    def _fetch(self, obj: RemoteObject) -> Path:
        path = self.cache_path(obj)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
        body = self._open(obj)
        try:
            with tmp.open("wb") as out:
                while chunk := body.read(_DOWNLOAD_CHUNK):
                    out.write(chunk)
            os.replace(tmp, path)
        except RemoteReadError:
            raise
        except Exception as e:
            raise RemoteReadError(f"{obj.uri}: download failed: {e}") from e
        finally:
            body.close()
            tmp.unlink(missing_ok=True)
        self._evict(keep=path)
        return path

    def _evict(self, keep: Path) -> None:
        """Delete least-recently-used cached files until the cache fits max_bytes (never ``keep``)."""
        files = []
        for p in self.cache_dir.glob("*/*"):
            if p.name.endswith(".part") or p == keep:
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            files.append((st.st_mtime, st.st_size, p))
        try:
            total = keep.stat().st_size + sum(size for _, size, _ in files)
        except FileNotFoundError:
            return
        for _, size, p in sorted(files, key=lambda f: f[0]):
            if total <= self.max_bytes:
                break
            p.unlink(missing_ok=True)
            try:
                p.parent.rmdir()
            except OSError:
                pass
            total -= size

    # ----------------------------------------------------------------------------------
    # Ahead-of-time downloads (parent only)
    # ----------------------------------------------------------------------------------

    def prefetch(self, objs: Sequence[RemoteObject]) -> None:
        """Start background downloads of objects that are neither cached nor in flight."""
        if not self.prefetch_enabled:
            return
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(_PREFETCH_THREADS, thread_name_prefix="trace-prefetch")
            for o in objs:
                if o.uri in self._inflight or self.cache_path(o).exists():
                    continue
                future = self._pool.submit(self._fetch, o)
                self._inflight[o.uri] = future
                future.add_done_callback(partial(self._prefetch_done, o.uri))

    def _prefetch_done(self, uri: str, future: Future[Path]) -> None:
        with self._lock:
            self._inflight.pop(uri, None)
        if future.exception() is not None:
            # Not fatal here: the load that needs this object retries and reports it.
            logger.debug(f"Prefetch of {uri} failed: {future.exception()}")

    def download_all(self, objs: Sequence[RemoteObject]) -> None:
        """Download every object not yet cached, concurrently, and wait for all of them."""
        todo = [o for o in objs if not self.cache_path(o).exists()]

        def fetch(o: RemoteObject) -> None:
            try:
                self._fetch(o)
            except RemoteReadError as e:
                logger.warning(f"Could not pre-download {o.uri}: {e}")

        with ThreadPoolExecutor(_PROBE_THREADS, thread_name_prefix="trace-download") as pool:
            list(pool.map(fetch, todo))


def _parse_first(line: bytes, obj: RemoteObject) -> Dict[str, Any]:
    try:
        parsed = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{obj.uri}: first line is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{obj.uri}: expected a JSON object, got {type(parsed).__name__}")
    return parsed
