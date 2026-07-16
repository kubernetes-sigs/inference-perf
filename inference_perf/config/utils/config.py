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
from typing import Optional

from pydantic import BaseModel, Field


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: Optional[str] = Field(
        default=None, description="HuggingFace model name or local path of the tokenizer to load."
    )
    trust_remote_code: Optional[bool] = Field(
        default=None, description="Allow the tokenizer to execute code from its repository when loading."
    )
    token: Optional[str] = Field(default=None, description="HuggingFace access token used to download the tokenizer.")
