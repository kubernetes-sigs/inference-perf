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
import queue
import re
import stat
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, TypedDict, TypeVar, Union

from inference_perf.datagen.replay.wire_trace_converter import _zstd

logger = logging.getLogger(__name__)

S3_SCHEME = "s3://"
# Covers at least one whole zstd block (up to 128 KiB), so the first range always decodes.
_PROBE_CHUNK = 256 * 1024
_DOWNLOAD_CHUNK = 1024 * 1024
_PROBE_THREADS = 16
_PREFETCH_THREADS = 8
# Wrong keys come back as these codes rather than as a missing-credentials error.
_CREDENTIAL_ERROR_CODES = frozenset(
    {"InvalidAccessKeyId", "SignatureDoesNotMatch", "AccessDenied", "ExpiredToken", "InvalidToken"}
)
_CREDENTIALS_HINT = (
    "set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (plus AWS_ENDPOINT_URL_S3 for a store other than "
    "AWS S3), or select a profile with AWS_PROFILE"
)
# Errors retrying cannot fix. Anything else (throttling, 5xx, resets, timeouts) is retried.
_PERMANENT_CODES = _CREDENTIAL_ERROR_CODES | {"NoSuchKey", "NoSuchBucket", "404", "PreconditionFailed", "412"}
# Waits before the second and third attempt of a read.
_RETRY_DELAYS = (0.5, 2.0)
# A first record longer than this is not probed further: the object is not line-delimited.
_PROBE_MAX = 256 * 1024 * 1024
# How often a process rereads the cache directory to see what other processes wrote.
_RESCAN_INTERVAL_S = 60.0
# A .part file this old belongs to a download that was cut off, not one in progress.
_STALE_PART_S = 3600.0

_T = TypeVar("_T")

# The name of a cache entry's directory: sha256(uri)[:24], then the ETag's alphanumerics.
# Only directories named like this are ever scanned or evicted.
_ENTRY_DIR = re.compile(r"[0-9a-f]{24}-[A-Za-z0-9]{0,40}")


class RemoteReadError(OSError):
    """An object could not be read. An OSError, so callers map it like any other read failure.

    ``code`` is the store's error code when there is one; ``retryable`` marks failures a
    later attempt may not hit (throttling, server errors, dropped connections).
    """

    def __init__(self, message: str, code: Optional[str] = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


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


# The slice of a boto3 S3 client this module uses. boto3 ships no types, so this states
# the contract instead; any S3-compatible client (or a test fake) satisfying it works.
class ListedObject(TypedDict):
    Key: str
    ETag: str
    Size: int


class ListPage(TypedDict, total=False):
    Contents: List[ListedObject]


class ObjectBody(Protocol):
    def read(self, amt: Optional[int] = None) -> bytes: ...

    def close(self) -> None: ...


class GetObjectOutput(TypedDict):
    Body: ObjectBody


class ListPaginator(Protocol):
    def paginate(self, *, Bucket: str, Prefix: str, Delimiter: str = ...) -> Iterable[ListPage]: ...


class S3Client(Protocol):
    def get_paginator(self, operation_name: str, /) -> ListPaginator: ...

    def get_object(self, *, Bucket: str, Key: str, IfMatch: str, Range: str = ...) -> GetObjectOutput: ...


# A first record, passed through untouched: callers inspect it, nothing here does.
TraceRecord = Dict[str, object]


def default_client() -> S3Client:
    import boto3
    from botocore.config import Config

    # One pooled connection per thread that may download at once, or urllib3 discards and reopens
    # them; and a short connect timeout, so a wrong endpoint fails in seconds rather than minutes.
    config = Config(max_pool_connections=_PROBE_THREADS + _PREFETCH_THREADS + 8, connect_timeout=10)
    client: S3Client = boto3.client("s3", config=config)
    return client


def check_not_all_denied(results: Sequence[Union[TraceRecord, None, Exception]]) -> None:
    """Fail with the credentials hint when every remote object was refused: the keys, not the data, are wrong.

    ``results`` holds one outcome per remote object. Any readable object, or any failure
    that is not a credential refusal, means the data is at fault, not the keys.
    """
    if results and all(isinstance(r, RemoteReadError) and r.code in _CREDENTIAL_ERROR_CODES for r in results):
        codes = sorted({r.code for r in results if isinstance(r, RemoteReadError) and r.code})
        raise ValueError(f"Every remote trace was refused ({', '.join(codes)}): {_CREDENTIALS_HINT}")


def prepare_private_dir(path: Path) -> None:
    """Create ``path`` owner-only, or accept it only if it already is: a real directory this user owns that others cannot write.

    For the default cache under the shared system temp dir, whose name another user could
    create first, as a symlink into this user's home or as a directory of planted files.
    """
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as e:
        raise ValueError(f"Cannot create the trace cache directory {path}: {e}") from e
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"Refusing trace cache directory {path}: it is a symlink; set remote_cache_dir")
    if not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"Refusing trace cache directory {path}: not a directory; set remote_cache_dir")
    getuid = getattr(os, "getuid", None)
    if getuid is not None and st.st_uid != getuid():
        raise ValueError(f"Refusing trace cache directory {path}: owned by uid {st.st_uid}; set remote_cache_dir")
    if st.st_mode & 0o022:
        raise ValueError(f"Refusing trace cache directory {path}: writable by other users; set remote_cache_dir")


class RemoteTraceStore:
    """Lists, probes and caches remote trace objects.

    Shared by the parent and every worker through a pickled copy. The copy rebuilds its
    client lazily, and never prefetches or evicts: only the parent knows the play order and
    holds the pins, so it alone decides what leaves the shared cache directory.

    Each process keeps an index of the cache directory (path -> last use, size), so
    eviction does not rescan the directory after every download; it rereads it at most
    every _RESCAN_INTERVAL_S to account for what other processes wrote.
    """

    def __init__(self, cache_dir: Path, max_bytes: int, client_factory: Optional[Callable[[], S3Client]] = None) -> None:
        self.cache_dir = cache_dir
        self.max_bytes = max_bytes
        # None looks default_client up at first use, so a replacement installed later applies.
        self.client_factory = client_factory
        self.prefetch_enabled = True
        self.evict_enabled = True
        # Off while the run is still being prepared, when downloads are expected rather than stalls.
        self.report_waits = True
        self.sync_downloads = 0
        self.sync_wait_s = 0.0
        self._init_runtime()

    def _init_runtime(self) -> None:
        self._client: Optional[S3Client] = None
        self._lock = threading.RLock()
        # uri -> set once its background download ends, whether or not it succeeded.
        self._inflight: Dict[str, threading.Event] = {}
        self._queue: Optional[queue.SimpleQueue[RemoteObject]] = None
        # Files eviction must keep, counted: the set being pre-downloaded or the download-ahead window.
        self._pinned: Counter[Path] = Counter()
        self._index: Optional[Dict[Path, Tuple[float, int]]] = None
        self._index_bytes = 0
        # Bytes of downloads in progress, reserved before they write so the cap covers them.
        self._reserved = 0
        self._scanned_at = 0.0

    def __getstate__(self) -> Dict[str, object]:
        state = dict(self.__dict__)
        for name in (
            "_client",
            "_lock",
            "_inflight",
            "_queue",
            "_pinned",
            "_index",
            "_index_bytes",
            "_reserved",
            "_scanned_at",
        ):
            state.pop(name, None)
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)
        self._init_runtime()
        self.prefetch_enabled = False
        self.evict_enabled = False
        self.sync_downloads = 0
        self.sync_wait_s = 0.0

    @property
    def client(self) -> S3Client:
        with self._lock:
            if self._client is None:
                self._client = (self.client_factory or default_client)()
            return self._client

    def check_writable(self) -> None:
        """Fail now, not one session at a time during the run, when the cache directory is unusable."""
        probe = self.cache_dir / f".write-check-{os.getpid()}"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as e:
            raise ValueError(f"remote_cache_dir {self.cache_dir} is not writable: {e}") from e

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
        # List from the longest literal prefix: whole segments, plus the start of the first wildcard one.
        literal: List[str] = []
        for segment in key.split("/"):
            if _is_magic(segment):
                literal.append(segment[: min(segment.index(c) for c in "*?[" if c in segment)])
                break
            literal.append(segment)
        prefix = "/".join(literal)
        return sorted((o for o in self._list(bucket, prefix) if match_key(key, o.key)), key=lambda o: o.key)

    def list_directory(self, uri: str) -> List[RemoteObject]:
        """Objects directly under a prefix named like ``*.json*``, as a local directory listing gives.

        The prefix is literal, as a local directory path is, and names starting with a dot match.
        """
        bucket, key = parse_s3_uri(uri)
        prefix = key.rstrip("/") + "/" if key.strip("/") else ""
        found = self._list(bucket, prefix, delimiter="/")
        return sorted((o for o in found if fnmatchcase(o.name, "*.json*")), key=lambda o: o.key)

    def _list(self, bucket: str, prefix: str, delimiter: str = "") -> List[RemoteObject]:
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError

        where = f"{S3_SCHEME}{bucket}/{prefix}"
        logger.info(f"Listing {where}")
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = (
                paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter)
                if delimiter
                else paginator.paginate(Bucket=bucket, Prefix=prefix)
            )
            objects: List[RemoteObject] = []
            for page in pages:
                for item in page.get("Contents", []):
                    if not item["Key"].endswith("/"):
                        objects.append(RemoteObject(bucket, item["Key"], item["ETag"], int(item["Size"])))
            return objects
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise ValueError(f"No credentials to list {where}: {_CREDENTIALS_HINT}") from e
        except ClientError as e:
            error = e.response.get("Error", {})
            hint = f"; {_CREDENTIALS_HINT}" if error.get("Code") in _CREDENTIAL_ERROR_CODES else ""
            raise ValueError(f"Cannot list {where}: {error.get('Code')} {error.get('Message')}{hint}") from e
        except BotoCoreError as e:
            raise ValueError(f"Cannot list {where}: {e}. Check AWS_ENDPOINT_URL_S3 or the profile's endpoint_url") from e

    # ----------------------------------------------------------------------------------
    # Reads
    # ----------------------------------------------------------------------------------

    def _open(self, obj: RemoteObject, byte_range: Optional[str] = None) -> ObjectBody:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            if byte_range is None:
                return self.client.get_object(Bucket=obj.bucket, Key=obj.key, IfMatch=obj.etag)["Body"]
            return self.client.get_object(Bucket=obj.bucket, Key=obj.key, IfMatch=obj.etag, Range=byte_range)["Body"]
        except ClientError as e:
            error = e.response.get("Error", {})
            code = str(error.get("Code"))
            if code in ("PreconditionFailed", "412"):
                raise RemoteReadError(f"{obj.uri} changed since it was listed", code=code) from e
            message = f"{obj.uri}: {code} {error.get('Message')}"
            raise RemoteReadError(message, code=code, retryable=code not in _PERMANENT_CODES) from e
        except BotoCoreError as e:
            raise RemoteReadError(f"{obj.uri}: {e}", retryable=True) from e

    def _retrying(self, attempt: Callable[[], _T], obj: RemoteObject) -> _T:
        """Run a read, retrying transient failures (throttling, 5xx, resets) a bounded number of times."""
        delays = _RETRY_DELAYS
        for tries in range(len(delays) + 1):
            try:
                return attempt()
            except RemoteReadError as e:
                if not e.retryable or tries == len(delays):
                    raise
                logger.info(f"Retrying {obj.uri} after: {e}")
                time.sleep(delays[tries])
        raise AssertionError("unreachable")

    def _read_range(self, obj: RemoteObject, start: int, end: int) -> bytes:
        body = self._open(obj, f"bytes={start}-{end}")
        try:
            # Read the ranged body to its end, so the connection goes back to the pool.
            chunks = []
            while chunk := body.read(_DOWNLOAD_CHUNK):
                chunks.append(chunk)
            return b"".join(chunks)
        except Exception as e:
            raise RemoteReadError(f"{obj.uri}: {e}", retryable=True) from e
        finally:
            body.close()

    def first_record(self, obj: RemoteObject) -> Optional[TraceRecord]:
        """Return the first non-blank line of a line-delimited object as a dict.

        Reads a 256 KiB range, then extends it fourfold while the first record does not
        fit, fetching only the new bytes, so only the start of the object is downloaded.
        """
        if obj.size == 0:
            return None  # a ranged read of an empty object is an error (416), not an empty body
        size = _PROBE_CHUNK
        data = b""
        while True:
            start = len(data)
            data += self._retrying(partial(self._read_range, obj, start, size - 1), obj)
            whole = len(data) < size or size >= obj.size
            text = _decompress_prefix(data, obj, whole) if obj.name.endswith(".zst") else data
            lines = text.split(b"\n")
            for line in lines[:-1]:
                if line.strip():
                    return _parse_first(line, obj)
            if whole:
                return _parse_first(lines[-1], obj) if lines[-1].strip() else None
            if size >= _PROBE_MAX:
                raise ValueError(f"{obj.uri}: no complete first record within {_PROBE_MAX} bytes")
            size *= 4

    def first_records(self, objs: Sequence[RemoteObject]) -> List[Union[TraceRecord, None, Exception]]:
        """first_record for each object, concurrently; a failure is returned in its slot."""

        def probe(o: RemoteObject) -> Union[TraceRecord, None, Exception]:
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
        it means the trace was not downloaded ahead in time. A background download that
        failed is retried here, so a transient error costs a wait, not a session.
        """
        with self._lock:
            pending = self._inflight.get(obj.uri)
        path = self.cache_path(obj)
        if pending is None and self._touch(path):
            return path
        started = time.perf_counter()
        reason = "it was not downloaded ahead"
        if pending is not None:
            pending.wait()
            reason = "its download ahead had not finished"
        if pending is None or not self._touch(path):
            if pending is not None:
                reason = "its download ahead failed"
            self._fetch(obj)
        waited = time.perf_counter() - started
        if self.report_waits:
            self.sync_downloads += 1
            self.sync_wait_s += waited
            # With no cache the parent's reads download by choice; a warning each time would be
            # noise. A worker's never are: the parent fetched the same file a moment before.
            if self.max_bytes > 0 or not self.evict_enabled:
                logger.warning(
                    f"Waited {waited:.2f}s for {obj.uri} ({obj.size} bytes): {reason} "
                    f"[{self.sync_downloads} waits, {self.sync_wait_s:.1f}s so far in this process]"
                )
        return path

    def _fetch(self, obj: RemoteObject) -> Path:
        path = self.cache_path(obj)
        with self._lock:
            # Reserve the object's size and make room first, so the download in progress
            # counts against the cap instead of landing on top of it.
            self._ensure_index()
            self._reserved += obj.size
            if self.evict_enabled:
                self._evict(keep=path)
        try:
            size = self._retrying(lambda: self._fetch_once(obj, path), obj)
        finally:
            with self._lock:
                self._reserved -= obj.size
        with self._lock:
            self._remember(path, size)
            if self.evict_enabled:
                self._evict(keep=path)
        return path

    def _fetch_once(self, obj: RemoteObject, path: Path) -> int:
        body = self._open(obj)
        tmp = path.parent / f".{os.getpid()}.{threading.get_ident()}.part"
        written = 0
        try:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                out = tmp.open("wb")
            except FileNotFoundError as e:  # another process evicted the entry's directory meanwhile
                raise RemoteReadError(f"{obj.uri}: cache entry removed during download", retryable=True) from e
            except OSError as e:
                raise RemoteReadError(f"{obj.uri}: cannot write to the cache: {e}") from e
            with out:
                while True:
                    try:
                        chunk = body.read(_DOWNLOAD_CHUNK)
                    except Exception as e:
                        raise RemoteReadError(f"{obj.uri}: download interrupted: {e}", retryable=True) from e
                    if not chunk:
                        break
                    try:
                        out.write(chunk)
                    except OSError as e:
                        raise RemoteReadError(f"{obj.uri}: cannot write to the cache: {e}") from e
                    written += len(chunk)
            try:
                os.replace(tmp, path)
            except FileNotFoundError as e:  # another process evicted the entry's directory meanwhile
                raise RemoteReadError(f"{obj.uri}: cache entry removed during download", retryable=True) from e
        finally:
            body.close()
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return written

    def _touch(self, path: Path) -> bool:
        """Mark a cached file as just used; False when it isn't cached."""
        try:
            os.utime(path)
        except OSError:  # missing, or the cache directory itself is unusable
            return False
        with self._lock:
            if self._index is not None and path in self._index:
                self._index[path] = (time.time(), self._index[path][1])
        return True

    def _scan(self) -> None:
        """Rebuild the index from disk, removing .part files of downloads that were cut off."""
        index: Dict[Path, Tuple[float, int]] = {}
        now = time.time()
        try:
            entries = list(os.scandir(self.cache_dir))
        except OSError:
            entries = []
        for entry in entries:
            # Anything not named like a cache entry is someone else's file: never touch it.
            if not _ENTRY_DIR.fullmatch(entry.name) or not entry.is_dir(follow_symlinks=False):
                continue
            try:
                files = list(os.scandir(entry.path))
            except OSError:
                continue
            for f in files:
                try:
                    st = f.stat(follow_symlinks=False)
                except OSError:
                    continue
                if f.name.endswith(".part"):
                    if now - st.st_mtime > _STALE_PART_S:
                        try:
                            os.unlink(f.path)
                        except OSError:
                            pass
                    continue
                index[Path(f.path)] = (st.st_mtime, st.st_size)
        self._index = index
        self._index_bytes = sum(size for _, size in index.values())
        self._scanned_at = time.monotonic()

    def _ensure_index(self) -> None:
        if self._index is None or time.monotonic() - self._scanned_at > _RESCAN_INTERVAL_S:
            self._scan()

    def _remember(self, path: Path, size: int) -> None:
        self._ensure_index()
        assert self._index is not None
        previous = self._index.get(path)
        self._index_bytes += size - (previous[1] if previous else 0)
        self._index[path] = (time.time(), size)

    def _evict(self, keep: Path) -> None:
        """Delete least-recently-used unpinned files until the cache fits max_bytes (never ``keep``).

        Pinned files count toward the total even though they are never deleted.
        """
        assert self._index is not None
        if self._index_bytes + self._reserved <= self.max_bytes:
            return
        candidates = sorted((used, p) for p, (used, _) in self._index.items() if p != keep and not self._pinned[p])
        for _, p in candidates:
            if self._index_bytes + self._reserved <= self.max_bytes:
                break
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:  # undeletable: leave it, and keep going
                logger.debug(f"Could not evict {p}: {e}")
                continue
            self._index_bytes -= self._index.pop(p)[1]
            try:
                p.parent.rmdir()
            except OSError:
                pass

    def pin(self, objs: Sequence[RemoteObject]) -> None:
        """Keep these objects from eviction until unpinned (counted, so overlapping sets compose)."""
        with self._lock:
            self._pinned.update(self.cache_path(o) for o in objs)

    def unpin(self, objs: Sequence[RemoteObject]) -> None:
        with self._lock:
            self._pinned.subtract(self.cache_path(o) for o in objs)
            self._pinned = +self._pinned

    # ----------------------------------------------------------------------------------
    # Ahead-of-time downloads (parent only)
    # ----------------------------------------------------------------------------------

    def prefetch(self, objs: Sequence[RemoteObject]) -> None:
        """Download objects that are neither cached nor in flight, in the background.

        The threads are daemons, so exiting never waits for downloads nobody needs any more;
        a download cut short leaves a .part file that a later scan removes.
        """
        if not self.prefetch_enabled:
            return
        with self._lock:
            if self._queue is None:
                self._queue = queue.SimpleQueue()
                for i in range(_PREFETCH_THREADS):
                    threading.Thread(target=self._prefetch_loop, name=f"trace-prefetch-{i}", daemon=True).start()
            for o in objs:
                if o.uri in self._inflight or self._touch(self.cache_path(o)):
                    continue
                self._inflight[o.uri] = threading.Event()
                self._queue.put(o)

    def _prefetch_loop(self) -> None:
        assert self._queue is not None
        while True:
            o = self._queue.get()
            try:
                self._fetch(o)
            except Exception as e:
                # Not fatal: the load that needs this object downloads it itself.
                logger.debug(f"Download ahead of {o.uri} failed: {e}")
            finally:
                with self._lock:
                    done = self._inflight.pop(o.uri, None)
                if done is not None:
                    done.set()

    def download_all(self, objs: Sequence[RemoteObject]) -> Dict[str, RemoteReadError]:
        """Download every object not yet cached, concurrently, and wait for all of them.

        Cached ones are touched. Pin the set first when it must survive its own downloads.
        Returns the failures by URI.
        """
        todo = [o for o in objs if not self._touch(self.cache_path(o))]
        failures: Dict[str, RemoteReadError] = {}

        def fetch(o: RemoteObject) -> None:
            try:
                self._fetch(o)
            except RemoteReadError as e:
                logger.warning(f"Could not pre-download {o.uri}: {e}")
                failures[o.uri] = e

        with ThreadPoolExecutor(_PROBE_THREADS, thread_name_prefix="trace-download") as pool:
            list(pool.map(fetch, todo))
        return failures


def _decompress_prefix(data: bytes, obj: RemoteObject, whole: bool) -> bytes:
    """Decompress the start of a .zst object, frame by frame.

    A prefix may end inside a frame; that is only an error when the prefix is the whole object.
    """
    zstd = _zstd()
    out = bytearray()
    decompressor = zstd.ZstdDecompressor()
    remaining = data
    try:
        while remaining:
            out += decompressor.decompress(remaining)
            if not decompressor.eof:
                break
            remaining = decompressor.unused_data
            decompressor = zstd.ZstdDecompressor()
    except zstd.ZstdError as e:
        raise RemoteReadError(f"{obj.uri}: corrupt .zst: {e}") from e
    if whole and remaining and not decompressor.eof:
        raise RemoteReadError(f"{obj.uri}: truncated .zst (ends inside a frame)")
    return bytes(out)


def _parse_first(line: bytes, obj: RemoteObject) -> TraceRecord:
    try:
        parsed = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{obj.uri}: first line is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{obj.uri}: expected a JSON object, got {type(parsed).__name__}")
    return parsed
