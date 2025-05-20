from abc import ABC, abstractmethod
from typing import Tuple

from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.config import ModelServerConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class LlmModelServerClient(ModelServerClient, ABC):
    @abstractmethod
    def __init__(self, config: ModelServerConfig, *args: Tuple[int, ...]) -> None:
        super().__init__(api_type=config.api.type)
        if config.model.tokenizer and config.model.tokenizer.pretrained_model_name_or_path:
            try:
                self.tokenizer = CustomTokenizer(config.model.tokenizer)
            except Exception as e:
                raise Exception("Tokenizer initialization failed") from e

        if config.load.api.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {config.load.api.type}")

        self.model_name = config.model.name
        self.uri = config.base_url
        self.load = config.load

    def get_tokenizer(self) -> CustomTokenizer:
        return self.tokenizer
