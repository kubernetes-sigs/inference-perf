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
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from inference_perf.config import CustomTokenizerConfig


class CustomTokenizer:
    def __init__(self, config: CustomTokenizerConfig):
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path, token=config.token, trust_remote_code=config.trust_remote_code
        )

    def count_tokens(self, text: str) -> int:
        if text == "":
            return 0
        return len(self.tokenizer(text).input_ids)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer
