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
from .base import DataGenerator, LazyLoadDataMixin
from inference_perf.config import DataGenType
from .mock_datagen import MockDataGenerator
from .hf_sharegpt_datagen import HFShareGPTDataGenerator
from .synthetic_datagen import SyntheticDataGenerator
from .random_datagen import RandomDataGenerator
from .shared_prefix_datagen import SharedPrefixDataGenerator
from .cnn_dailymail_datagen import CNNDailyMailDataGenerator
from .infinity_instruct_datagen import InfinityInstructDataGenerator
from .hf_billsum_datagen import BillsumConversationsDataGenerator
from typing import Type


def get_datagen_class(datagen_type: DataGenType) -> Type[DataGenerator]:
    if datagen_type == DataGenType.Mock:
        return MockDataGenerator
    elif datagen_type == DataGenType.Random:
        return RandomDataGenerator
    elif datagen_type == DataGenType.Synthetic:
        return SyntheticDataGenerator
    elif datagen_type == DataGenType.ShareGPT:
        return HFShareGPTDataGenerator
    elif datagen_type == DataGenType.SharedPrefix:
        return SharedPrefixDataGenerator
    elif datagen_type == DataGenType.CNNDailyMail:
        return CNNDailyMailDataGenerator
    elif datagen_type == DataGenType.InfinityInstruct:
        return InfinityInstructDataGenerator
    elif datagen_type == DataGenType.BillsumConversations:
        return BillsumConversationsDataGenerator
    raise ValueError(f"Unknown data generator type: {datagen_type}")


__all__ = [
    "DataGenerator",
    "LazyLoadDataMixin",
    "MockDataGenerator",
    "HFShareGPTDataGenerator",
    "SyntheticDataGenerator",
    "RandomDataGenerator",
    "SharedPrefixDataGenerator",
    "CNNDailyMailDataGenerator",
    "InfinityInstructDataGenerator",
    "BillsumConversationsDataGenerator",
    "get_datagen_class",
]
