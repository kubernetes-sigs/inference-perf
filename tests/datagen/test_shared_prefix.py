import pytest
from unittest.mock import Mock

from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.apis.user_session import UserSessionCompletionAPIData
from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig


def create_mock_tokenizer() -> Mock:
    """Create a mock tokenizer for testing."""
    mock_tokenizer = Mock()
    mock_hf_tokenizer = Mock()
    mock_hf_tokenizer.vocab_size = 32000
    mock_hf_tokenizer.decode = Mock(side_effect=lambda ids, **kwargs: f"text_{len(ids)}")
    mock_hf_tokenizer.batch_decode = Mock(side_effect=lambda ids_list, **kwargs: [f"text_{len(ids)}" for ids in ids_list])
    mock_tokenizer.get_tokenizer.return_value = mock_hf_tokenizer
    return mock_tokenizer


def create_api_config(api_type: APIType) -> APIConfig:
    """Create an APIConfig for testing."""
    return APIConfig(type=api_type)


def create_data_config(
    num_groups: int = 2,
    num_prompts_per_group: int = 3,
    system_prompt_len: int = 10,
    question_len: int = 5,
    output_len: int = 20,
    enable_multi_turn_chat: bool = False,
) -> DataConfig:
    """Create a DataConfig with shared_prefix settings for testing."""
    config = DataConfig()
    config.shared_prefix = Mock()
    config.shared_prefix.num_groups = num_groups
    config.shared_prefix.num_prompts_per_group = num_prompts_per_group
    config.shared_prefix.system_prompt_len = system_prompt_len
    config.shared_prefix.question_len = question_len
    config.shared_prefix.output_len = output_len
    config.shared_prefix.enable_multi_turn_chat = enable_multi_turn_chat
    config.shared_prefix.question_distribution = None
    config.shared_prefix.output_distribution = None
    return config


class TestSharedPrefixDataGeneratorBasic:
    """Basic tests for SharedPrefixDataGenerator."""

    def test_get_supported_apis(self) -> None:
        """Test that both Completion and Chat APIs are supported."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config()
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        supported = generator.get_supported_apis()
        assert APIType.Completion in supported
        assert APIType.Chat in supported

    def test_prompts_count(self) -> None:
        """Test that correct number of prompts are generated."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(num_groups=2, num_prompts_per_group=3)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        assert len(generator.prompts) == 6  # 2 groups * 3 prompts
        assert len(generator.prompt_pairs) == 6

    def test_prompt_pairs_structure(self) -> None:
        """Test that prompt_pairs contain (shared_prefix, question) tuples."""
        api_config = create_api_config(APIType.Chat)
        data_config = create_data_config()
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        for shared_prefix, question in generator.prompt_pairs:
            assert isinstance(shared_prefix, str)
            assert isinstance(question, str)


class TestSharedPrefixCompletionAPI:
    """Tests for Completion API support."""

    def test_load_lazy_data_returns_completion_api_data(self) -> None:
        """Test that load_lazy_data returns CompletionAPIData for Completion API."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)

        result = generator.load_lazy_data(lazy_data)

        assert isinstance(result, CompletionAPIData)
        assert result.max_tokens == generator.flat_output_lens[0]

    @pytest.mark.asyncio
    async def test_completion_api_to_payload(self) -> None:
        """Test that CompletionAPIData generates correct payload."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(output_len=50, enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)
        result = generator.load_lazy_data(lazy_data)

        payload = await result.to_payload("test-model", 100, False, True)

        assert payload["model"] == "test-model"
        assert "prompt" in payload
        assert payload["max_tokens"] == 50
        assert payload["stream"] is True

    def test_completion_api_prompt_content(self) -> None:
        """Test that CompletionAPIData prompt matches prompts list."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)

        result = generator.load_lazy_data(lazy_data)

        assert isinstance(result, CompletionAPIData)
        assert result.prompt == generator.prompts[0]

    def test_get_data_yields_lazy_load_data_for_completion(self) -> None:
        """Test that get_data yields LazyLoadInferenceAPIData which resolves to CompletionAPIData."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        data_gen = generator.get_data()

        first_lazy = next(data_gen)
        assert isinstance(first_lazy, LazyLoadInferenceAPIData)

        first_item = generator.load_lazy_data(first_lazy)
        assert isinstance(first_item, CompletionAPIData)


class TestSharedPrefixChatAPI:
    """Tests for Chat API support."""

    def test_load_lazy_data_returns_chat_api_data(self) -> None:
        """Test that load_lazy_data returns ChatCompletionAPIData for Chat API."""
        api_config = create_api_config(APIType.Chat)
        data_config = create_data_config(enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)

        result = generator.load_lazy_data(lazy_data)

        assert isinstance(result, ChatCompletionAPIData)
        assert result.max_tokens == generator.flat_output_lens[0]

    def test_chat_api_messages_structure(self) -> None:
        """Test that Chat API messages have system and user roles."""
        api_config = create_api_config(APIType.Chat)
        data_config = create_data_config(enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)

        result = generator.load_lazy_data(lazy_data)

        assert isinstance(result, ChatCompletionAPIData)
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"

    def test_chat_api_messages_content_from_prompt_pairs(self) -> None:
        """Test that Chat API messages content matches prompt_pairs."""
        api_config = create_api_config(APIType.Chat)
        data_config = create_data_config(enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)

        result = generator.load_lazy_data(lazy_data)
        expected_shared_prefix, expected_question = generator.prompt_pairs[0]

        assert isinstance(result, ChatCompletionAPIData)
        assert result.messages[0].content == expected_shared_prefix
        assert result.messages[1].content == expected_question

    @pytest.mark.asyncio
    async def test_chat_api_to_payload(self) -> None:
        """Test that ChatCompletionAPIData generates correct payload with system and user messages."""
        api_config = create_api_config(APIType.Chat)
        data_config = create_data_config(output_len=50, enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)
        result = generator.load_lazy_data(lazy_data)

        payload = await result.to_payload("test-model", 100, False, False)

        assert payload["model"] == "test-model"
        assert "messages" in payload
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["max_tokens"] == 50
        assert payload["stream"] is False

    def test_get_data_yields_lazy_load_data_for_chat(self) -> None:
        """Test that get_data yields LazyLoadInferenceAPIData which resolves to ChatCompletionAPIData."""
        api_config = create_api_config(APIType.Chat)
        data_config = create_data_config(enable_multi_turn_chat=False)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        data_gen = generator.get_data()

        first_lazy = next(data_gen)
        assert isinstance(first_lazy, LazyLoadInferenceAPIData)

        first_item = generator.load_lazy_data(first_lazy)
        assert isinstance(first_item, ChatCompletionAPIData)
        assert len(first_item.messages) == 2


class TestSharedPrefixMultiTurn:
    """Tests for multi-turn chat support."""

    def test_multi_turn_creates_user_sessions(self) -> None:
        """Test that multi-turn mode creates user sessions."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(num_groups=2, num_prompts_per_group=3, enable_multi_turn_chat=True)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        assert len(generator.user_sessions) == 6  # 2 groups * 3 prompts

    def test_multi_turn_load_lazy_data_returns_user_session_data(self) -> None:
        """Test that multi-turn load_lazy_data returns UserSessionCompletionAPIData."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(enable_multi_turn_chat=True)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        lazy_data = LazyLoadInferenceAPIData(data_index=0, prefered_worker_id=0)

        result = generator.load_lazy_data(lazy_data)

        assert isinstance(result, UserSessionCompletionAPIData)

    def test_multi_turn_get_data_yields_lazy_load_data(self) -> None:
        """Test that multi-turn get_data yields LazyLoadInferenceAPIData."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config(enable_multi_turn_chat=True)
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        data_gen = generator.get_data()

        first_item = next(data_gen)
        assert isinstance(first_item, LazyLoadInferenceAPIData)


class TestSharedPrefixValidation:
    """Tests for validation and error handling."""

    def test_requires_tokenizer(self) -> None:
        """Test that tokenizer is required."""
        api_config = create_api_config(APIType.Completion)
        data_config = create_data_config()

        with pytest.raises((ValueError, AttributeError)):
            SharedPrefixDataGenerator(api_config, data_config, None)

    def test_requires_shared_prefix_config(self) -> None:
        """Test that shared_prefix config is required."""
        api_config = create_api_config(APIType.Completion)
        data_config = DataConfig()
        data_config.shared_prefix = None
        tokenizer = create_mock_tokenizer()

        with pytest.raises(ValueError, match="Shared Prefix config is required"):
            SharedPrefixDataGenerator(api_config, data_config, tokenizer)
