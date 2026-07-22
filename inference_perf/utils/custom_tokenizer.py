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
import logging
import threading

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER
from inference_perf.config import CustomTokenizerConfig

logger = logging.getLogger(__name__)


class CustomTokenizer:
    def __init__(self, config: CustomTokenizerConfig) -> None:
        self.tokenizer: PreTrainedTokenizerBase = _load_tokenizer_with_deadline(config)

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        if text == "":
            return 0

        # add_special_tokens=False is for counting a fragment of a larger sequence (e.g. a single
        # streamed chunk), where prepending a BOS per call would over-count by one token per fragment.
        # Some tokenizers don't set model_max_length which defaults to VERY_LARGE_INTEGER.
        # Prevent overflow and log spam by skipping truncation.
        if self.tokenizer.model_max_length == VERY_LARGE_INTEGER:
            return len(self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids)
        return len(
            self.tokenizer(
                text, truncation=True, max_length=self.tokenizer.model_max_length, add_special_tokens=add_special_tokens
            ).input_ids
        )

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer


def _load_tokenizer_with_deadline(config: CustomTokenizerConfig) -> PreTrainedTokenizerBase:
    # AutoTokenizer.from_pretrained downloads tokenizer files from Hugging Face Hub when they
    # are not cached, and a wedged transfer (e.g. hf_xet hanging on a CDN error instead of
    # raising) would otherwise block the whole benchmark forever with no visible failure.
    # Loading in a daemon thread bounds the wait; the stuck thread cannot be cancelled, but it
    # no longer blocks the process from failing fast or exiting.
    result: dict[str, PreTrainedTokenizerBase] = {}
    error: dict[str, BaseException] = {}

    def load() -> None:
        try:
            result["tokenizer"] = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                config.pretrained_model_name_or_path, token=config.token, trust_remote_code=config.trust_remote_code
            )
        except BaseException as e:
            error["error"] = e

    logger.info("Loading tokenizer '%s'", config.pretrained_model_name_or_path)
    thread = threading.Thread(target=load, name="tokenizer-load", daemon=True)
    thread.start()
    # load_timeout=None must reach join() untouched: join(timeout=None) waits
    # indefinitely, which is how null disables the deadline. Coercing None to a
    # number (join(timeout=0) returns immediately) would fail every load.
    thread.join(timeout=config.load_timeout)
    if thread.is_alive():
        raise TimeoutError(
            f"Loading tokenizer '{config.pretrained_model_name_or_path}' did not finish within "
            f"{config.load_timeout} seconds. This usually means the download from Hugging Face Hub "
            "is stuck (network issues or a Hub/CDN outage). Check connectivity to huggingface.co, "
            "try HF_HUB_DISABLE_XET=1 to surface the underlying download error, pre-populate the "
            "HF cache, or point 'tokenizer.pretrained_model_name_or_path' at a local directory. "
            "The deadline is configurable via 'tokenizer.load_timeout' (null disables it)."
        )
    if "error" in error:
        raise error["error"]
    return result["tokenizer"]
