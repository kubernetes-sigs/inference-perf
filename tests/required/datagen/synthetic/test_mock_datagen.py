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
from inference_perf.apis import EmbeddingsAPIData, RerankAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, EmbeddingsConfig, RerankConfig
from inference_perf.datagen.synthetic.mock_datagen import MockDataGenerator


def test_mock_datagen_embeddings_single_input() -> None:
    generator = MockDataGenerator(APIConfig(type=APIType.Embeddings), DataConfig(type=DataGenType.Mock), None)

    data = next(generator.get_data())

    assert isinstance(data, EmbeddingsAPIData)
    assert data.input == "mock prompt 1-0"


def test_mock_datagen_embeddings_batch_and_options() -> None:
    api_config = APIConfig(type=APIType.Embeddings, embeddings=EmbeddingsConfig(batch_size=3, dimensions=64))
    generator = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)

    data = next(generator.get_data())

    assert isinstance(data, EmbeddingsAPIData)
    assert data.input == ["mock prompt 1-0", "mock prompt 1-1", "mock prompt 1-2"]
    assert data.dimensions == 64


def test_mock_datagen_rerank_default_document_count() -> None:
    generator = MockDataGenerator(APIConfig(type=APIType.Rerank), DataConfig(type=DataGenType.Mock), None)

    data = next(generator.get_data())

    assert isinstance(data, RerankAPIData)
    assert data.query == "mock query 1"
    assert len(data.documents) == 10
    assert data.documents[0] == "mock document 1-0"


def test_mock_datagen_rerank_document_count_and_options() -> None:
    api_config = APIConfig(type=APIType.Rerank, rerank=RerankConfig(document_count=3, top_n=2))
    generator = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)

    data = next(generator.get_data())

    assert isinstance(data, RerankAPIData)
    assert data.query == "mock query 1"
    assert data.documents == ["mock document 1-0", "mock document 1-1", "mock document 1-2"]
    assert data.top_n == 2
