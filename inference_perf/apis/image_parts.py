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

"""Image content parts in the shapes traces record them and requests send them.

- OTel GenAI parts: ``{"type": "blob", "modality": "image", "mime_type": ..., "content": <base64>}``,
  ``uri`` and ``file`` parts. Trace ingest normalizes every recorded image to one of these.
- OpenAI content parts: ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}``,
  the shape the replay graph and requests carry.
- Anthropic ``image`` blocks, recorded by Anthropic traces and sent by the Anthropic renderer.
"""

import hashlib
import re
from typing import Any, Dict, Optional, Tuple


def split_data_url(url: str) -> Optional[Tuple[str, str]]:
    """``(mime_type, base64_data)`` of a base64 data URL, or None for any other URL."""
    if url.startswith("data:") and ";base64," in url:
        header, data = url.split(",", 1)
        return header[len("data:") :].split(";", 1)[0], data
    return None


# What :func:`summarize_base64` leaves in place of a payload, e.g. in inference-perf's own spans.
_SUMMARY = re.compile(r"\[\d+ bytes, sha256:[0-9a-f]{16}\]")


def base64_size(data: str) -> int:
    """Decoded size in bytes of a base64 payload, computed without decoding it."""
    return len(data) * 3 // 4 - data[-2:].count("=")


def summarize_base64(data: str) -> str:
    """A short stand-in for a base64 payload: its decoded size and a hash of the payload."""
    return f"[{base64_size(data)} bytes, sha256:{hashlib.sha256(data.encode('ascii', 'replace')).hexdigest()[:16]}]"


def is_image_part(part: Any) -> bool:
    """Whether a content part is an image, in any of the shapes above."""
    if not isinstance(part, dict):
        return False
    if part.get("type") in ("blob", "uri"):
        return bool(part.get("modality", "image") == "image")
    if part.get("type") == "file":
        # A Chat Completions ``file`` part (e.g. a PDF) has no modality: only an image file counts.
        return bool(part.get("modality") == "image")
    return part.get("type") in ("image_url", "input_image", "image")


def anthropic_image_to_part(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic ``image`` block to an OTel ``blob``, ``uri`` or ``file`` part."""
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64":
        return {"type": "blob", "modality": "image", "mime_type": source.get("media_type", ""), "content": source.get("data")}
    if source.get("type") == "url":
        return {"type": "uri", "modality": "image", "uri": source.get("url")}
    if source.get("type") == "file":
        return {"type": "file", "modality": "image", "file_id": source.get("file_id")}
    return None


def openai_image_to_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI ``image_url`` / ``input_image`` part to an OTel part.

    An image without a URL (an uploaded ``file_id``, or a redacted URL) becomes a ``file``
    part: its bytes are not in the trace, so it counts as missing.
    """
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    detail = image_url.get("detail") if isinstance(image_url, dict) else part.get("detail")
    data_url = split_data_url(url) if isinstance(url, str) else None
    result: Dict[str, Any]
    if data_url:
        result = {"type": "blob", "modality": "image", "mime_type": data_url[0], "content": data_url[1]}
    elif isinstance(url, str) and url:
        result = {"type": "uri", "modality": "image", "uri": url}
    else:
        result = {"type": "file", "modality": "image", "file_id": part.get("file_id")}
    if detail:
        result["detail"] = detail
    return result


def to_otel_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """An image part in any of the shapes above, as an OTel ``blob`` / ``uri`` / ``file`` part."""
    if part.get("type") == "image":
        return anthropic_image_to_part(part) or {}
    if part.get("type") in ("image_url", "input_image"):
        return openai_image_to_part(part)
    return part


def image_bytes(part: Dict[str, Any]) -> Optional[str]:
    """An OTel part's base64 image payload, or None when it holds no image bytes (or only a summary)."""
    content = part.get("content")
    if part.get("type") == "blob" and isinstance(content, str) and content and not _SUMMARY.fullmatch(content):
        return content
    return None


def to_openai_image(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert an image part in any of the shapes above to an OpenAI ``image_url`` part.

    Returns None when the part carries no image to send.
    """
    part = to_otel_part(part)
    data = image_bytes(part)
    if data is not None:
        url = f"data:{part.get('mime_type') or 'image/png'};base64,{data}"
    elif part.get("type") == "uri" and part.get("uri"):
        url = part["uri"]
    else:
        return None
    image_url = {"url": url}
    if part.get("detail"):
        image_url["detail"] = part["detail"]
    return {"type": "image_url", "image_url": image_url}


def image_digest(part: Dict[str, Any]) -> str:
    """A short, stable identity for an image part (its bytes or URL, and detail), for comparing messages without its bytes."""
    part = to_otel_part(part)
    source = image_bytes(part) or (part.get("uri") if part.get("type") == "uri" else None) or ""
    digest = hashlib.sha256(f"{part.get('detail') or ''}|".encode("utf-8") + source.encode("ascii", "replace"))
    return "[image:" + digest.hexdigest()[:16] + "]"


# Text a tracing tool leaves where it dropped an image: Claude Code's native OTel tool
# output, and Langfuse's media reference token.
_IMAGE_PLACEHOLDER = "[image]"
_LANGFUSE_MEDIA = "@@@langfuseMedia:"


def count_missing_images(value: Any) -> int:
    """Count images a trace no longer carries: image parts without bytes (or with only a
    :func:`summarize_base64` summary), and known placeholders.

    Tool-call arguments are skipped, since they are model output rather than content.
    """
    if isinstance(value, str):
        return 1 if value.strip() == _IMAGE_PLACEHOLDER else value.count(_LANGFUSE_MEDIA)
    if isinstance(value, list):
        return sum(count_missing_images(item) for item in value)
    if not isinstance(value, dict):
        return 0
    if is_image_part(value):
        return 0 if to_openai_image(value) else 1
    return sum(count_missing_images(item) for key, item in value.items() if key != "arguments")
