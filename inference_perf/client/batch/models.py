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
"""The OpenAI Batch API objects the batch client reads.

Field names follow the OpenAI API reference, which llm-d-batch-gateway serves too. Fields not declared here
are ignored.
"""

from typing import Optional

from pydantic import BaseModel, Field


class BatchFile(BaseModel):
    """The file object POST /v1/files returns."""

    id: str


class BatchRequestCounts(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0


class BatchError(BaseModel):
    """One reason a batch failed validation, such as a malformed input line."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    line: Optional[int] = None


class BatchErrors(BaseModel):
    data: list[BatchError] = []


class Batch(BaseModel):
    """The batch object /v1/batches returns. Timestamps are Unix seconds on the server's clock."""

    id: str
    status: str
    endpoint: str
    input_file_id: str
    completion_window: str
    created_at: int
    in_progress_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    expires_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    request_counts: BatchRequestCounts = Field(default_factory=BatchRequestCounts)
    errors: Optional[BatchErrors] = None


class BatchLineResponse(BaseModel):
    """The HTTP response the server got for one input line."""

    status_code: int
    request_id: Optional[str] = None
    body: Optional[dict[str, object]] = None


class BatchLineError(BaseModel):
    """Why one input line got no response, such as batch_expired."""

    code: Optional[str] = None
    message: Optional[str] = None


class BatchOutputLine(BaseModel):
    """One line of an output or error file: the answer to the input line with the same custom_id."""

    custom_id: str
    response: Optional[BatchLineResponse] = None
    error: Optional[BatchLineError] = None
