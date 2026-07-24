import json
import pytest
from inference_perf.apis import CompletionAPIData
from inference_perf.datagen.hf_sharegpt_datagen import HFShareGPTDataGenerator


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
