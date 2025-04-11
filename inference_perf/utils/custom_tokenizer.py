# Copyright 2025 The Kubernetes Authors.
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
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class CustomTokenizer:
    def __init__(self, tokenizer_id: str, token: Optional[str], trust_remote_code: Optional[bool]):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=token, trust_remote_code=trust_remote_code)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer(text).input_ids)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer
