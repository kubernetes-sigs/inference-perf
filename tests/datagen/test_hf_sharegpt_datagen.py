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

import unittest
from unittest.mock import MagicMock, patch


class TestHFShareGPTDataGenerator(unittest.TestCase):
    """Tests for HFShareGPTDataGenerator, specifically the StopIteration handling."""

    @patch("inference_perf.datagen.hf_sharegpt_datagen.load_dataset")
    def test_next_data_reloads_on_stopiteration(self, mock_load_dataset: MagicMock) -> None:
        """Test that _next_data reloads the dataset when StopIteration is raised."""
        from inference_perf.datagen.hf_sharegpt_datagen import HFShareGPTDataGenerator
        from inference_perf.config import APIConfig, APIType, DataConfig

        # Create mock data items
        mock_data_item_1 = {"conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi"}]}
        mock_data_item_2 = {"conversations": [{"from": "human", "value": "Bye"}, {"from": "gpt", "value": "Goodbye"}]}

        # First iterator: returns one item then raises StopIteration
        first_iterator = iter([mock_data_item_1])
        # Second iterator (after reload): returns another item
        second_iterator = iter([mock_data_item_2])

        # Mock load_dataset to return different iterators on each call
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset.__iter__ = MagicMock(side_effect=[first_iterator, second_iterator])

        # Create the data generator
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig()
        generator = HFShareGPTDataGenerator(api_config, data_config, tokenizer=None)

        # First call should return the first item
        result1 = generator._next_data()
        self.assertEqual(result1["conversations"][0]["value"], "Hello")

        # Second call should trigger StopIteration, reload, and return second item
        result2 = generator._next_data()
        self.assertEqual(result2["conversations"][0]["value"], "Bye")

        # Verify load_dataset was called twice (initial load + reload)
        self.assertEqual(mock_load_dataset.call_count, 2)

    @patch("inference_perf.datagen.hf_sharegpt_datagen.load_dataset")
    def test_next_data_normal_iteration(self, mock_load_dataset: MagicMock) -> None:
        """Test that _next_data returns items normally when not exhausted."""
        from inference_perf.datagen.hf_sharegpt_datagen import HFShareGPTDataGenerator
        from inference_perf.config import APIConfig, APIType, DataConfig

        # Create mock data items
        mock_data_items = [
            {"conversations": [{"from": "human", "value": "First"}, {"from": "gpt", "value": "Response 1"}]},
            {"conversations": [{"from": "human", "value": "Second"}, {"from": "gpt", "value": "Response 2"}]},
            {"conversations": [{"from": "human", "value": "Third"}, {"from": "gpt", "value": "Response 3"}]},
        ]

        # Mock load_dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset.__iter__ = MagicMock(return_value=iter(mock_data_items))

        # Create the data generator
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig()
        generator = HFShareGPTDataGenerator(api_config, data_config, tokenizer=None)

        # Iterate through items
        result1 = generator._next_data()
        result2 = generator._next_data()
        result3 = generator._next_data()

        self.assertEqual(result1["conversations"][0]["value"], "First")
        self.assertEqual(result2["conversations"][0]["value"], "Second")
        self.assertEqual(result3["conversations"][0]["value"], "Third")

        # load_dataset should only be called once (initial load)
        self.assertEqual(mock_load_dataset.call_count, 1)


if __name__ == "__main__":
    unittest.main()
