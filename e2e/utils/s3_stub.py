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

"""An in-process S3 stub for e2e tests: the S3 API surface boto3 uses to read replay traces.

ListObjectsV2 (pagination, Delimiter, EncodingType=url) and GetObject (Range -> 206/416,
If-Match -> 412), served path-style on 127.0.0.1, with every request recorded. Signatures
are not verified; a wrong access key is refused with InvalidAccessKeyId. Stdlib only.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, unquote, urlsplit
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class Seen:
    """One request the stub answered."""

    method: str
    bucket: str
    key: str  # empty for a listing
    query: Dict[str, str]
    headers: Dict[str, str]  # lower-cased names
    status: int


@dataclass
class S3Stub:
    """Serves ``objects`` (keyed by (bucket, key)) until the ``with`` block ends.

    ``access_key``: refuse requests signed with any other key. ``full_gets_once``: refuse a
    second full (non-ranged) GET of an object with AccessDenied, so a process that misses a
    cache it should have hit fails loudly instead of quietly downloading again.
    """

    objects: Dict[Tuple[str, str], bytes] = field(default_factory=dict)
    page_size: int = 2
    access_key: Optional[str] = None
    full_gets_once: bool = False
    requests: List[Seen] = field(default_factory=list)
    _served_whole: Set[Tuple[str, str]] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _server: Optional[ThreadingHTTPServer] = None

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.objects[(bucket, key)] = data

    def etag(self, bucket: str, key: str) -> str:
        return '"' + hashlib.md5(self.objects[(bucket, key)]).hexdigest() + '"'

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def gets(self, key: Optional[str] = None) -> List[Seen]:
        """The GetObject requests, all or for one key."""
        return [r for r in self.requests if r.key and (key is None or r.key == key)]

    def __enter__(self) -> "S3Stub":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()


def _handler(stub: S3Stub) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, as boto3's connection pool expects

        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, status: int, body: bytes, headers: Dict[str, str]) -> None:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, code: str, message: str) -> int:
            body = (
                f"<?xml version='1.0' encoding='UTF-8'?><Error><Code>{code}</Code><Message>{escape(message)}</Message></Error>"
            )
            self._send(status, body.encode(), {"Content-Type": "application/xml"})
            return status

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            # Path-style only (/bucket/key): what botocore sends to an IP endpoint.
            bucket, _, key = unquote(parts.path).lstrip("/").partition("/")
            query = {k: v[0] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
            headers = {k.lower(): v for k, v in self.headers.items()}
            if stub.access_key is not None and f"Credential={stub.access_key}/" not in headers.get("authorization", ""):
                status = self._error(
                    403, "InvalidAccessKeyId", "The AWS Access Key Id you provided does not exist in our records."
                )
            elif key:
                status = self._get(bucket, key, headers)
            else:
                status = self._list(bucket, query)
            with stub._lock:
                stub.requests.append(Seen(self.command, bucket, key, query, headers, status))

        def _list(self, bucket: str, q: Dict[str, str]) -> int:
            if q.get("list-type") != "2":
                return self._error(400, "InvalidRequest", "only ListObjectsV2 is supported")
            if not any(b == bucket for b, _ in stub.objects):
                return self._error(404, "NoSuchBucket", bucket)
            prefix, delimiter = q.get("prefix", ""), q.get("delimiter", "")
            encoded = q.get("encoding-type") == "url"  # botocore asks for it, and decodes keys when echoed back

            def enc(s: str) -> str:
                return quote(s, safe="/") if encoded else s

            after = q.get("continuation-token", "")
            max_keys = min(int(q.get("max-keys", "1000")), stub.page_size)
            # Entries in key order: objects, and with a delimiter the rolled-up common prefixes.
            entries: List[Tuple[str, bool]] = []
            for b, k in sorted(stub.objects):
                if b != bucket or not k.startswith(prefix):
                    continue
                rest = k[len(prefix) :]
                if delimiter and delimiter in rest:
                    common = prefix + rest[: rest.index(delimiter) + len(delimiter)]
                    if not entries or entries[-1] != (common, True):
                        entries.append((common, True))
                else:
                    entries.append((k, False))
            entries = [e for e in entries if e[0] > after]
            page, truncated = entries[:max_keys], len(entries) > max_keys
            xml = [
                "<?xml version='1.0' encoding='UTF-8'?>",
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
            ]
            xml.append(f"<Name>{escape(bucket)}</Name><Prefix>{escape(enc(prefix))}</Prefix>")
            if delimiter:
                xml.append(f"<Delimiter>{escape(enc(delimiter))}</Delimiter>")
            if encoded:
                xml.append("<EncodingType>url</EncodingType>")
            xml.append(f"<MaxKeys>{max_keys}</MaxKeys><KeyCount>{len(page)}</KeyCount>")
            xml.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
            if after:
                xml.append(f"<ContinuationToken>{escape(after)}</ContinuationToken>")
            if truncated:
                xml.append(f"<NextContinuationToken>{escape(page[-1][0])}</NextContinuationToken>")
            for name, is_prefix in page:
                if is_prefix:
                    xml.append(f"<CommonPrefixes><Prefix>{escape(enc(name))}</Prefix></CommonPrefixes>")
                else:
                    xml.append(
                        f"<Contents><Key>{escape(enc(name))}</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>"
                        f"<ETag>{escape(stub.etag(bucket, name))}</ETag><Size>{len(stub.objects[(bucket, name)])}</Size>"
                        "<StorageClass>STANDARD</StorageClass></Contents>"
                    )
            xml.append("</ListBucketResult>")
            self._send(200, "".join(xml).encode(), {"Content-Type": "application/xml"})
            return 200

        def _get(self, bucket: str, key: str, headers: Dict[str, str]) -> int:
            data = stub.objects.get((bucket, key))
            if data is None:
                return self._error(404, "NoSuchKey", key)
            etag = stub.etag(bucket, key)
            if headers.get("if-match") not in (None, etag, "*"):
                return self._error(412, "PreconditionFailed", "At least one of the pre-conditions you specified did not hold")
            base = {"ETag": etag, "Last-Modified": "Thu, 01 Jan 2026 00:00:00 GMT", "Accept-Ranges": "bytes"}
            byte_range = headers.get("range")
            if byte_range is None:
                if stub.full_gets_once:
                    # Check and claim together, so two concurrent full GETs cannot both be served.
                    with stub._lock:
                        first = (bucket, key) not in stub._served_whole
                        stub._served_whole.add((bucket, key))
                    if not first:
                        return self._error(403, "AccessDenied", f"{key} was already downloaded once")
                self._send(200, data, base)
                return 200
            start_s, _, end_s = byte_range.removeprefix("bytes=").partition("-")
            start, end = int(start_s), min(int(end_s) if end_s else len(data) - 1, len(data) - 1)
            if start >= len(data):
                return self._error(416, "InvalidRange", "The requested range is not satisfiable")
            self._send(206, data[start : end + 1], {**base, "Content-Range": f"bytes {start}-{end}/{len(data)}"})
            return 206

    return Handler
