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
import json
import math
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from inference_perf.config import CustomTokenizerConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# If the deadline mechanism is disarmed (e.g. a bare thread.join()), the hung-download
# test blocks forever; fail the file instead of hanging CI.
pytestmark = pytest.mark.timeout(30)


class TestCustomTokenizerLoadDeadline(unittest.TestCase):
    @patch("inference_perf.utils.custom_tokenizer.AutoTokenizer")
    def test_load_success(self, mock_auto_tokenizer: MagicMock) -> None:
        fake_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = fake_tokenizer

        tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="some/model", load_timeout=5.0))

        self.assertIs(tokenizer.get_tokenizer(), fake_tokenizer)
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("some/model", token=None, trust_remote_code=None)

    @patch("inference_perf.utils.custom_tokenizer.AutoTokenizer")
    def test_load_error_propagates(self, mock_auto_tokenizer: MagicMock) -> None:
        mock_auto_tokenizer.from_pretrained.side_effect = OSError("repo not found")

        with self.assertRaises(OSError):
            CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="some/model", load_timeout=5.0))

    @patch("inference_perf.utils.custom_tokenizer.AutoTokenizer")
    def test_hung_download_raises_timeout(self, mock_auto_tokenizer: MagicMock) -> None:
        # Simulate hf_xet wedging mid-download: from_pretrained never returns.
        release = threading.Event()

        def hang(*args: Any, **kwargs: Any) -> MagicMock:
            release.wait()
            return MagicMock()

        mock_auto_tokenizer.from_pretrained.side_effect = hang
        try:
            start = time.monotonic()
            with self.assertRaises(TimeoutError) as ctx:
                CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="some/model", load_timeout=0.1))
            elapsed = time.monotonic() - start
            self.assertIn("did not finish within 0.1 seconds", str(ctx.exception))
            # The timeout must fire near the configured deadline, not merely eventually.
            self.assertLess(elapsed, 5.0)
        finally:
            release.set()

    @patch("inference_perf.utils.custom_tokenizer.AutoTokenizer")
    def test_null_load_timeout_disables_deadline(self, mock_auto_tokenizer: MagicMock) -> None:
        # load_timeout=null must mean "no deadline": the load is waited on until
        # it completes, however long it takes. A regression that coerces None to
        # a number (e.g. thread.join(timeout=0)) would return with the loader
        # thread still alive and raise a spurious TimeoutError on every load, so
        # the slow load below must still succeed.
        fake_tokenizer = MagicMock()

        def slow_load(*args: Any, **kwargs: Any) -> MagicMock:
            time.sleep(0.2)
            return fake_tokenizer

        mock_auto_tokenizer.from_pretrained.side_effect = slow_load

        tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="some/model", load_timeout=None))

        self.assertIs(tokenizer.get_tokenizer(), fake_tokenizer)
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("some/model", token=None, trust_remote_code=None)

    def test_default_load_timeout(self) -> None:
        self.assertEqual(CustomTokenizerConfig().load_timeout, 300.0)


class TestLoadTimeoutValidation(unittest.TestCase):
    def test_accepts_positive_and_null(self) -> None:
        self.assertEqual(CustomTokenizerConfig(load_timeout=300.0).load_timeout, 300.0)
        self.assertEqual(CustomTokenizerConfig(load_timeout=0.001).load_timeout, 0.001)
        self.assertIsNone(CustomTokenizerConfig(load_timeout=None).load_timeout)

    def test_rejects_non_positive_and_non_finite(self) -> None:
        # 0 and negatives would make thread.join() return immediately, so every
        # load would surface as a Hub outage; inf and nan raise from
        # threading internals, bypassing the TimeoutError path. All four must
        # fail config validation instead.
        for value in (0, -5, math.inf, math.nan):
            with self.assertRaises(ValidationError, msg=f"load_timeout={value}"):
                CustomTokenizerConfig(load_timeout=value)


# A word-level tokenizer (vocabulary "a", "b") loaded from tmp_path, with model_max_length 4.
# `truncation` is written into tokenizer.json as shipped (None: disabled).
def _word_tokenizer(tmp_path: Path, truncation: Optional[dict[str, Any]] = None) -> CustomTokenizer:
    spec: dict[str, Any] = {
        "version": "1.0",
        "truncation": truncation,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"<unk>": 0, "a": 1, "b": 2}, "unk_token": "<unk>"},
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4}), encoding="utf-8"
    )
    return CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(tmp_path)))


# 3 tokens count as 3; 8 tokens are capped at model_max_length (4), the same count
# Hugging Face gives when truncating to it.
def test_count_is_capped_at_model_max_length(tmp_path: Path) -> None:
    tokenizer = _word_tokenizer(tmp_path)
    hf = tokenizer.get_tokenizer()

    assert tokenizer.count_tokens("a b a") == 3
    assert tokenizer.count_tokens("a b a b a b a b") == 4
    assert len(hf("a b a b a b a b", truncation=True, max_length=4).input_ids) == 4


# count_tokens must not reconfigure the tokenizer's backend: the backend is shared by
# every thread using the tokenizer (with open loop, the datagen thread encodes with
# truncation=False while the event loop counts), so a call that enabled truncation
# could make another thread's encode come back cut to model_max_length. After counting
# an over-long text the backend still has truncation disabled.
def test_count_tokens_leaves_the_backend_untouched(tmp_path: Path) -> None:
    tokenizer = _word_tokenizer(tmp_path)
    backend = tokenizer.get_tokenizer().backend_tokenizer

    tokenizer.count_tokens("a b a b a b a b")

    assert backend.truncation is None


# A tokenizer.json that ships with truncation enabled (max_length 2) has it switched
# off at load, so no later call has to reconfigure the backend; counts are still
# capped at model_max_length (4), not at the shipped 2.
def test_shipped_truncation_is_disabled_at_load(tmp_path: Path) -> None:
    shipped = {"direction": "Right", "max_length": 2, "strategy": "LongestFirst", "stride": 0}
    tokenizer = _word_tokenizer(tmp_path, truncation=shipped)

    assert tokenizer.get_tokenizer().backend_tokenizer.truncation is None
    assert tokenizer.count_tokens("a b a b a b a b") == 4


if __name__ == "__main__":
    unittest.main()
