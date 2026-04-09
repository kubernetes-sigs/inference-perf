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

import unittest
from unittest.mock import Mock
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.config import APIConfig, DataConfig, APIType, DataGenType, SharedPrefix, Distribution


class TestSharedPrefixDatagen(unittest.TestCase):
    def test_shared_prefix_with_distribution(self) -> None:
        mock_tokenizer = Mock()
        mock_tokenizer_obj = Mock()
        mock_tokenizer_obj.decode.return_value = "test prompt"
        mock_tokenizer_obj.batch_decode.side_effect = lambda x, skip_special_tokens=True: [
            f"question {i}" for i in range(len(x))
        ]
        mock_tokenizer_obj.vocab_size = 1000
        mock_tokenizer.get_tokenizer.return_value = mock_tokenizer_obj

        api_config = APIConfig(type=APIType.Completion)
        shared_prefix_cfg = SharedPrefix(
            num_groups=2,
            num_prompts_per_group=3,
            system_prompt_len=10,
            question_len=20,
            output_len=30,
            question_distribution=Distribution(min=10, max=20, mean=15, std_dev=2),
            output_distribution=Distribution(min=20, max=30, mean=25, std_dev=2),
        )
        data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)

        datagen = SharedPrefixDataGenerator(api_config=api_config, config=data_config, tokenizer=mock_tokenizer)

        self.assertEqual(len(datagen.prompts), 6)
        self.assertEqual(len(datagen.flat_output_lens), 6)

        # Verify output lengths are within bounds
        for length in datagen.flat_output_lens:
            self.assertGreaterEqual(length, 20)
            self.assertLessEqual(length, 30)

    def test_shared_prefix_with_expression(self) -> None:
        mock_tokenizer = Mock()
        mock_tokenizer_obj = Mock()
        mock_tokenizer_obj.decode.return_value = "test prompt"
        mock_tokenizer_obj.batch_decode.side_effect = lambda x, skip_special_tokens=True: [
            f"question {i}" for i in range(len(x))
        ]
        mock_tokenizer_obj.vocab_size = 1000
        mock_tokenizer.get_tokenizer.return_value = mock_tokenizer_obj

        api_config = APIConfig(type=APIType.Completion)
        shared_prefix_cfg = SharedPrefix(
            num_groups=2,
            num_prompts_per_group=3,
            system_prompt_len=10,
            question_len=20,
            output_len=30,
            question_distribution="Normal(15, 2)",
            output_distribution="Normal(25, 2)",
        )
        data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)

        datagen = SharedPrefixDataGenerator(api_config=api_config, config=data_config, tokenizer=mock_tokenizer)

        self.assertEqual(len(datagen.prompts), 6)
        self.assertEqual(len(datagen.flat_output_lens), 6)

        # For normal distribution, we can't strictly assert bounds unless we clamp,
        # but we can check they are ints and not crash.
        for length in datagen.flat_output_lens:
            self.assertIsInstance(length, int)
