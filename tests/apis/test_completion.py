from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIType


def test_completion_api_data() -> None:
    data = CompletionAPIData(prompt="Hello, world!")
    assert data.get_api_type() == APIType.Completion
    assert data.prompt == "Hello, world!"
    assert data.to_payload("test-model", 100, False, True) == {
        "model": "test-model",
        "prompt": "Hello, world!",
        "max_tokens": 100,
        "ignore_eos": False,
        "stream": True,
    }
