import json
import pytest
from unittest.mock import MagicMock
from inference_perf.apis import CompletionAPIData, TemplateAPIData
from inference_perf.config import APIConfig, APIType, TemplateConfig, TemplateResponseConfig
from inference_perf.datagen.dataset.hf_sharegpt_datagen import HFShareGPTDataGenerator


def test_get_conversation_turn_content_dict() -> None:
    # We bypass __init__ to avoid actually loading the dataset
    generator = HFShareGPTDataGenerator.__new__(HFShareGPTDataGenerator)
    generator.data_key = "conversations"
    generator.content_key = "value"

    data = {"conversations": [{"from": "human", "value": "madoka"}, {"from": "gpt", "value": "magika"}]}

    assert generator.get_conversation_turn_content(data, 0) == "madoka"
    assert generator.get_conversation_turn_content(data, 1) == "magika"


def test_get_conversation_turn_content_json_string() -> None:
    generator = HFShareGPTDataGenerator.__new__(HFShareGPTDataGenerator)
    generator.data_key = "conversations"
    generator.content_key = "value"

    # https://github.com/kubernetes-sigs/inference-perf/issues/429:
    # The dataset sometimes contains a string containing a JSON
    # object rather than the object itself for some reason.
    data = {
        "conversations": [
            json.dumps({"from": "human", "value": "madoka"}),
            json.dumps({"from": "gpt", "value": "magika"}),
        ]
    }

    assert generator.get_conversation_turn_content(data, 0) == "madoka"
    assert generator.get_conversation_turn_content(data, 1) == "magika"


def test_get_conversation_turn_content_unsupported_type() -> None:
    generator = HFShareGPTDataGenerator.__new__(HFShareGPTDataGenerator)
    generator.data_key = "conversations"
    generator.content_key = "value"

    data = {
        "conversations": [
            123,  # Invalid type
            ["madoka"],  # Invalid type
        ]
    }

    with pytest.raises(Exception, match="Conversation from upstream gave unsupported type: int"):
        generator.get_conversation_turn_content(data, 0)

    with pytest.raises(Exception, match="Conversation from upstream gave unsupported type: list"):
        generator.get_conversation_turn_content(data, 1)


def test_get_anthropic_messages_data_rejects_unexpected_chat_data() -> None:
    generator = HFShareGPTDataGenerator.__new__(HFShareGPTDataGenerator)
    # Deliberately stub the instance past its real types: the test only needs
    # get_anthropic_messages_data to reject a non-chat payload, so the tokenizer is never used
    # and get_chat_data is replaced with a generator of the wrong element type on purpose.
    generator.tokenizer = object()  # type: ignore[assignment]
    generator.get_chat_data = lambda: iter([CompletionAPIData(prompt="hello")])  # type: ignore[method-assign,assignment,return-value]

    with pytest.raises(Exception, match="Expected ChatCompletionAPIData, got CompletionAPIData"):
        next(generator.get_anthropic_messages_data())


def test_completion_prompt_sent_through_a_template() -> None:
    generator = HFShareGPTDataGenerator.__new__(HFShareGPTDataGenerator)
    template = TemplateConfig(route="/generate", body={"text": "${prompt}"}, response=TemplateResponseConfig(text_path="text"))
    generator.api_config = APIConfig(type=APIType.Template, template=template)
    generator.data_key = "conversations"
    generator.content_key = "value"
    generator.min_num_turns = 2
    generator.input_distribution = None
    generator.output_distribution = None
    generator._dataset_ready = True
    generator.sharegpt_dataset = iter(
        [{"conversations": [{"from": "human", "value": "madoka"}, {"from": "gpt", "value": "magika"}]}]
    )
    tokenizer = MagicMock()
    tokenizer.get_tokenizer.return_value.encode.return_value = [1]
    tokenizer.count_tokens.return_value = 1
    generator.tokenizer = tokenizer

    data = next(generator.get_data())

    assert isinstance(data, TemplateAPIData)
    assert data.prompt == "madoka"
    assert data.max_tokens == 1
