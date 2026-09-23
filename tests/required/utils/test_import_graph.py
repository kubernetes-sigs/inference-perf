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
"""Config must be able to import from under ``inference_perf.utils``.

``inference_perf.utils.__init__`` imports the tokenizer eagerly. If the
tokenizer imported ``inference_perf.config`` at runtime, then any config module
importing e.g. ``inference_perf.utils.numeric.expression`` would close a cycle
and fail at interpreter start. The tokenizer therefore takes its config class
as a type-only import; this pins that.
"""

import subprocess
import sys

import inference_perf.utils.custom_tokenizer as custom_tokenizer


# The tokenizer module has no runtime binding named CustomTokenizerConfig: the import is type-only.
def test_tokenizer_does_not_import_config_at_runtime() -> None:
    assert not hasattr(custom_tokenizer, "CustomTokenizerConfig")


# A fresh interpreter can import inference_perf.config first and then a utils submodule; exit code 0.
def test_config_then_utils_import_order_is_clean() -> None:
    code = "import inference_perf.config; import inference_perf.utils.numeric.expression"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
