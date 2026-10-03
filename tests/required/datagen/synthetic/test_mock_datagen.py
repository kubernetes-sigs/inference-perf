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
from inference_perf.apis import EmbeddingsAPIData, TemplateAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, EmbeddingsConfig, TemplateConfig
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


def test_mock_datagen_template() -> None:
    template = TemplateConfig(route="/generate", body={"text": "${prompt}"}, text_path="text")
    generator = MockDataGenerator(APIConfig(type=APIType.Template, template=template), DataConfig(type=DataGenType.Mock), None)

    data = next(generator.get_data())

    assert isinstance(data, TemplateAPIData)
    assert data.prompt == "1 2 3 1"
    assert data.template == template
