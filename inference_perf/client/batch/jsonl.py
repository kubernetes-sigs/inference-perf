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
"""Writes records as an OpenAI Batch API input file, one JSONL line per record.

Each line's body is the request the record's own API shape sends on its own, built with streaming off,
because a batch answers with whole responses.
"""

import json
from typing import Sequence

from pydantic import BaseModel

from inference_perf.apis import InferenceAPIData
from inference_perf.apis.user_session import UserSessionCompletionAPIData


class BatchInput(BaseModel):
    """The input file for one batch, ready to upload."""

    # The route every line posts to, and so the endpoint the batch is created against.
    endpoint: str
    # Line i's custom_id. The output and error files name the line they answer by it.
    custom_ids: list[str]
    content: bytes


async def build_batch_input(
    records: Sequence[InferenceAPIData],
    *,
    custom_id_prefix: str,
    model_name: str,
    max_tokens: int,
    ignore_eos: bool,
) -> BatchInput:
    """Build one batch's input file from `records`, in order.

    Line i gets the custom_id `{custom_id_prefix}-{i}`. All records must use one route, because a batch
    runs against a single endpoint. A user-session turn and a trace replay event are refused: each builds its
    prompt from earlier responses, and a batch returns none of them until it ends. A user-session turn would
    wait forever, and a replay event would go out with its recorded placeholder text. A replay event is known
    by the hook the load generator awaits before sending it, so this module needs no datagen import.
    """
    if not records:
        raise ValueError("A batch needs at least one record")
    endpoint = records[0].get_route()
    custom_ids: list[str] = []
    lines: list[bytes] = []
    for index, record in enumerate(records):
        if isinstance(record, UserSessionCompletionAPIData):
            raise ValueError(f"Record {index} is a turn of user session {record.user_session_id}, which a batch cannot hold")
        elif hasattr(record, "wait_for_predecessors_and_substitute"):
            raise ValueError(f"Record {index} waits on earlier responses that a batch does not return")
        route = record.get_route()
        if route != endpoint:
            raise ValueError(f"All records in a batch must use one route: record {index} uses {route}, not {endpoint}")
        custom_id = f"{custom_id_prefix}-{index}"
        body = await record.to_request_body(
            effective_model_name=model_name, max_tokens=max_tokens, ignore_eos=ignore_eos, streaming=False
        )
        line = {"custom_id": custom_id, "method": "POST", "url": route, "body": body}
        custom_ids.append(custom_id)
        lines.append(json.dumps(line).encode() + b"\n")
    return BatchInput(endpoint=endpoint, custom_ids=custom_ids, content=b"".join(lines))
