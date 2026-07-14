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
import threading
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from inference_perf.config import CustomTokenizerConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer


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
            with self.assertRaises(TimeoutError) as ctx:
                CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="some/model", load_timeout=0.1))
            self.assertIn("did not finish within 0.1 seconds", str(ctx.exception))
        finally:
            release.set()

    def test_default_load_timeout(self) -> None:
        self.assertEqual(CustomTokenizerConfig().load_timeout, 300.0)


if __name__ == "__main__":
    unittest.main()
