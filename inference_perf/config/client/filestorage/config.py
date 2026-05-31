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

from pydantic import BaseModel


class StorageConfigBase(BaseModel):
    path: str = f"reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_file_prefix: Optional[str] = None


class GoogleCloudStorageConfig(StorageConfigBase):
    bucket_name: str


class SimpleStorageServiceConfig(StorageConfigBase):
    bucket_name: str
    endpoint_url: Optional[str] = None
    region_name: Optional[str] = None
    addressing_style: Optional[Literal["auto", "virtual", "path"]] = None


class StorageConfig(BaseModel):
    local_storage: StorageConfigBase = StorageConfigBase()
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = None
    simple_storage_service: Optional[SimpleStorageServiceConfig] = None
