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
import pickle
from pathlib import Path
from typing import Optional, Type, cast

import pytest

from inference_perf.config import APIConfig, APIType
from inference_perf.config.datagen.config import DataConfig, DataGenType
from inference_perf.datagen.base import DataGenerator
from inference_perf.datagen.cnn_dailymail_datagen import CNNDailyMailDataGenerator
from inference_perf.datagen.hf_billsum_datagen import BillsumConversationsDataGenerator
from inference_perf.datagen.hf_sharegpt_datagen import HFShareGPTDataGenerator
from inference_perf.datagen.infinity_instruct_datagen import InfinityInstructDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# Non-None, picklable stand-in. The streaming datagens only check the tokenizer
# for None at construction; the pickle round-trip below never touches it.
_DUMMY_TOKENIZER = cast(CustomTokenizer, object())

_RECORDS = [
    {"id": "1", "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
    {"id": "2", "conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}]},
]


@pytest.fixture
def dataset_path(tmp_path: Path) -> str:
    path = tmp_path / "data.json"
    path.write_text(json.dumps(_RECORDS))
    return str(path)


@pytest.mark.parametrize(
    "cls, datagen_type, iterator_attr, tokenizer",
    [
        (HFShareGPTDataGenerator, DataGenType.ShareGPT, "sharegpt_dataset", None),
        (CNNDailyMailDataGenerator, DataGenType.CNNDailyMail, "cnn_dailymail_dataset", _DUMMY_TOKENIZER),
        (BillsumConversationsDataGenerator, DataGenType.BillsumConversations, "billsum_dataset", None),
        (InfinityInstructDataGenerator, DataGenType.InfinityInstruct, "infinity_instruct_dataset", None),
    ],
)
def test_streaming_datagen_is_picklable(
    dataset_path: str,
    cls: Type[DataGenerator],
    datagen_type: DataGenType,
    iterator_attr: str,
    tokenizer: Optional[CustomTokenizer],
) -> None:
    """Streaming datagens must survive a pickle round-trip.

    forkserver/spawn load-generator workers pickle the Worker process (and its
    datagen) at start. The live HuggingFace streaming iterator is generator-backed
    and not picklable, so each streaming datagen drops it on pickle and rebuilds it
    on unpickle. Without that, .start() raises 'cannot pickle generator object'.
    """
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(type=datagen_type, path=dataset_path)

    generator = cls(api_config, data_config, tokenizer)
    assert iterator_attr in generator.__dict__

    # __getstate__ must omit the unpicklable streaming iterator.
    state = generator.__getstate__()
    assert isinstance(state, dict)
    assert iterator_attr not in state

    # The exact round-trip forkserver performs must succeed and rebuild the iterator.
    restored = pickle.loads(pickle.dumps(generator))
    assert iterator_attr in restored.__dict__
