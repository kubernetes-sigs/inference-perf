from .base import DataGenerator, InferenceData
from typing import Generator
from datasets import load_dataset

class HFShareGPTDataGenerator(DataGenerator):
    def __init__(self, dataset_file: str = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json") -> None:
        #Initialize the data generator by loading the ShareGPT dataset.
        self.sharegpt_dataset = load_dataset("json", data_files=dataset_file)["train"]

    def get_data(self) -> Generator[InferenceData, None, None]:
        #Stream system prompts from ShareGPT conversations.
        for data in self.sharegpt_dataset:
            if "conversations" in data and len(data["conversations"]) >= 2: # Filter out the conversations with less than 2 turns.
                yield InferenceData(system_prompt=data["conversations"][0]["value"])  # First message as system prompt.