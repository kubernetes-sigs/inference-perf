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
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class StorageConfigBase(BaseModel):
    path: str = Field(
        default=f"reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        description="Directory or object key prefix where report files are written."
        " '{timestamp}' is replaced with the run's start time.",
    )
    report_file_prefix: Optional[str] = Field(default=None, description="Prefix added to every report file name.")


class GoogleCloudStorageConfig(StorageConfigBase):
    bucket_name: str = Field(description="Name of the Google Cloud Storage bucket where reports are uploaded.")


class SimpleStorageServiceConfig(StorageConfigBase):
    bucket_name: str = Field(description="Name of the S3 bucket where reports are uploaded.")
    endpoint_url: Optional[str] = Field(default=None, description="Custom endpoint URL, for S3-compatible object stores.")
    region_name: Optional[str] = Field(default=None, description="AWS region of the bucket.")
    addressing_style: Optional[Literal["auto", "virtual", "path"]] = Field(
        default=None, description="S3 addressing style: 'auto', 'virtual' (bucket in hostname) or 'path'."
    )


class StorageConfig(BaseModel):
    local_storage: StorageConfigBase = Field(default=StorageConfigBase(), description="Save reports to the local filesystem.")
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = Field(
        default=None, description="Upload reports to Google Cloud Storage."
    )
    simple_storage_service: Optional[SimpleStorageServiceConfig] = Field(
        default=None, description="Upload reports to S3 or an S3-compatible object store."
    )
