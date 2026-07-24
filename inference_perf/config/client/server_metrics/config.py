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
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


class PrometheusClientConfig(BaseModel):
    scrape_interval: int = Field(default=15, description="Scrape interval of the Prometheus server in seconds.")
    url: Optional[HttpUrl] = Field(default=None, description="URL of the Prometheus server to query.")
    filters: List[str] = Field(
        default=[], description="PromQL label matchers (e.g. 'namespace=\"default\"') applied to every metric query."
    )
    google_managed: bool = Field(
        default=False, description="Query Google Cloud Managed Service for Prometheus instead of a self-hosted server."
    )

    @model_validator(mode="after")
    def check_exclusive_fields(self) -> "PrometheusClientConfig":
        if bool(self.url) == bool(self.google_managed):
            raise ValueError("Exactly one of 'url' or 'google_managed' must be set.")
        return self
