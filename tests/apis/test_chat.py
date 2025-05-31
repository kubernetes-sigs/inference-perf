from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIType


def test_chat_completion_api_data() -> None:
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="Hello, world!")])
    assert data.get_api_type() == APIType.Chat
    assert len(data.messages) == 1
    assert data.to_payload("test-model", 100, False, False) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_tokens": 100,
        "ignore_eos": False,
    }
