"""Tests for SharedPrefixDataGenerator multi-turn chat functionality."""

import pytest
from unittest.mock import Mock
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.apis.user_session import UserSessionChatAPIData, UserSessionCompletionAPIData
from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, SharedPrefix


from typing import Any


def create_mock_tokenizer(vocab_size: int = 1000) -> Mock:
    """Create a mock tokenizer for testing."""
    mock_tokenizer = Mock()
    mock_hf_tokenizer = Mock()
    mock_hf_tokenizer.vocab_size = vocab_size

    # Make decode return a deterministic string based on token IDs
    def mock_decode(token_ids: Any, skip_special_tokens: bool = True) -> str:
        return f"text_{hash(tuple(token_ids)) % 10000}"

    mock_hf_tokenizer.decode = mock_decode
    mock_tokenizer.get_tokenizer.return_value = mock_hf_tokenizer
    return mock_tokenizer


class TestSharedPrefixMultiTurn:
    """Tests for multi-turn chat with SharedPrefixDataGenerator."""

    def test_user_session_has_group_id(self) -> None:
        """Test that user sessions are created with correct group_id."""
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=3,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        # Should have 3 groups * 2 prompts = 6 user sessions
        assert len(generator.user_sessions) == 6

        # Check that group_ids are assigned correctly
        # Sessions 0, 1 should be group 0
        # Sessions 2, 3 should be group 1
        # Sessions 4, 5 should be group 2
        assert generator.user_sessions[0].group_id == 0
        assert generator.user_sessions[1].group_id == 0
        assert generator.user_sessions[2].group_id == 1
        assert generator.user_sessions[3].group_id == 1
        assert generator.user_sessions[4].group_id == 2
        assert generator.user_sessions[5].group_id == 2

    def test_questions_by_group_structure(self) -> None:
        """Test that questions_by_group is correctly populated."""
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=3,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        # Should have 2 groups
        assert len(generator.questions_by_group) == 2
        assert len(generator.shared_prefixes_by_group) == 2

        # Each group should have 3 questions
        assert len(generator.questions_by_group[0]) == 3
        assert len(generator.questions_by_group[1]) == 3

    def test_load_lazy_data_uses_correct_group_questions_chat(self) -> None:
        """Test that load_lazy_data selects questions from the correct group for Chat API."""
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=3,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        # Total 6 user sessions (2 groups * 3 prompts)
        num_users = len(generator.user_sessions)
        assert num_users == 6

        # Test round 0: each user gets question from their group
        for user_id in range(num_users):
            lazy_data = LazyLoadInferenceAPIData(data_index=user_id, prefered_worker_id=0)
            result = generator.load_lazy_data(lazy_data)

            assert isinstance(result, UserSessionChatAPIData)

            user_session = generator.user_sessions[user_id]
            group_id = user_session.group_id
            expected_question = generator.questions_by_group[group_id][0]  # round 0 -> question 0

            # The user message should contain the question from the correct group
            user_message = next(m for m in result.messages if m.role == "user")
            assert user_message.content == expected_question

    def test_load_lazy_data_different_questions_per_round(self) -> None:
        """Test that different rounds get different questions from the same group."""
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=3,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        num_users = len(generator.user_sessions)
        user_id = 0  # First user (group 0)
        group_id = generator.user_sessions[user_id].group_id

        questions_per_round = []
        for round_num in range(3):  # 3 rounds
            data_index = round_num * num_users + user_id
            lazy_data = LazyLoadInferenceAPIData(data_index=data_index, prefered_worker_id=0)
            result = generator.load_lazy_data(lazy_data)

            user_message = next(m for m in result.messages if m.role == "user")
            questions_per_round.append(user_message.content)

            # Verify question comes from the correct group
            expected_question = generator.questions_by_group[group_id][round_num % 3]
            assert user_message.content == expected_question

        # All 3 questions should be different (from the same group)
        assert len(set(questions_per_round)) == 3

    def test_load_lazy_data_completion_api_question_only(self) -> None:
        """Test that Completion API multi-turn passes only the question (not shared_prefix)."""
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        user_id = 0
        group_id = generator.user_sessions[user_id].group_id

        lazy_data = LazyLoadInferenceAPIData(data_index=user_id, prefered_worker_id=0)
        result = generator.load_lazy_data(lazy_data)

        assert isinstance(result, UserSessionCompletionAPIData)

        # The prompt should be only the question, not shared_prefix + question
        expected_question = generator.questions_by_group[group_id][0]
        assert result.prompt == expected_question

        # Verify shared_prefix is NOT in the prompt (it's in user_session.context)
        shared_prefix = generator.shared_prefixes_by_group[group_id]
        assert shared_prefix not in result.prompt

    def test_shared_prefix_consistency_across_rounds(self) -> None:
        """Test that the same user always uses the same shared_prefix across rounds."""
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=3,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        num_users = len(generator.user_sessions)

        for user_id in range(num_users):
            user_session = generator.user_sessions[user_id]
            group_id = user_session.group_id
            expected_shared_prefix = generator.shared_prefixes_by_group[group_id]

            # Check multiple rounds
            for round_num in range(5):
                data_index = round_num * num_users + user_id
                lazy_data = LazyLoadInferenceAPIData(data_index=data_index, prefered_worker_id=0)
                result = generator.load_lazy_data(lazy_data)

                # System message should always have the same shared_prefix
                system_message = next(m for m in result.messages if m.role == "system")
                assert system_message.content == expected_shared_prefix

    def test_user_sessions_not_shuffled(self) -> None:
        """Test that user_sessions maintain their original order (not shuffled)."""
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=3,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        # User session IDs should be in order
        for i, session in enumerate(generator.user_sessions):
            assert session.user_session_id == f"user_session_{i}"

    def test_context_variable_naming(self) -> None:
        """Test that LocalUserSession uses 'context' (not 'contexts')."""
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=1,
                num_prompts_per_group=1,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        session = generator.user_sessions[0]

        # Should have 'context' attribute (not 'contexts')
        assert hasattr(session, "context")
        # 'contexts' should NOT exist as an instance attribute
        assert "contexts" not in session.__dict__


class TestSharedPrefixSingleTurn:
    """Tests for single-turn (non-multi-turn) SharedPrefixDataGenerator."""

    def test_single_turn_completion_api(self) -> None:
        """Test single-turn mode with Completion API."""
        from inference_perf.apis.completion import CompletionAPIData

        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=3,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=False,  # Single-turn
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        # Should have 2 * 3 = 6 prompts
        assert len(generator.prompts) == 6
        assert len(generator.prompt_pairs) == 6

        # No user sessions for single-turn
        assert len(generator.user_sessions) == 0

        # Test get_data() returns CompletionAPIData
        data_gen = generator.get_data()
        result = next(data_gen)
        assert isinstance(result, CompletionAPIData)
        assert result.prompt in generator.prompts

    def test_single_turn_chat_api(self) -> None:
        """Test single-turn mode with Chat API."""
        from inference_perf.apis.chat import ChatCompletionAPIData

        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=3,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=False,  # Single-turn
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        # Should have 2 * 3 = 6 prompts
        assert len(generator.prompts) == 6
        assert len(generator.prompt_pairs) == 6

        # No user sessions for single-turn
        assert len(generator.user_sessions) == 0

        # Test get_data() returns ChatCompletionAPIData
        data_gen = generator.get_data()
        result = next(data_gen)
        assert isinstance(result, ChatCompletionAPIData)

        # Should have system and user messages
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"

        # Verify the pair exists in prompt_pairs
        found = False
        for shared_prefix, question in generator.prompt_pairs:
            if result.messages[0].content == shared_prefix and result.messages[1].content == question:
                found = True
                break
        assert found, "Message pair not found in prompt_pairs"

    def test_single_turn_get_request_by_index_completion(self) -> None:
        """Test get_request_by_index for Completion API."""
        from inference_perf.apis.completion import CompletionAPIData

        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=False,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        for i in range(len(generator.prompts)):
            result = generator.get_request_by_index(i)
            assert isinstance(result, CompletionAPIData)
            assert result.prompt == generator.prompts[i]

    def test_single_turn_get_request_by_index_chat(self) -> None:
        """Test get_request_by_index for Chat API."""
        from inference_perf.apis.chat import ChatCompletionAPIData

        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=False,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        for i in range(len(generator.prompts)):
            result = generator.get_request_by_index(i)
            assert isinstance(result, ChatCompletionAPIData)

            shared_prefix, question = generator.prompt_pairs[i]
            assert result.messages[0].content == shared_prefix
            assert result.messages[1].content == question

    def test_single_turn_load_lazy_data_completion(self) -> None:
        """Test load_lazy_data for single-turn Completion API."""
        from inference_perf.apis.completion import CompletionAPIData
        from inference_perf.apis.base import LazyLoadInferenceAPIData

        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=False,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        for i in range(len(generator.prompts)):
            lazy_data = LazyLoadInferenceAPIData(data_index=i, prefered_worker_id=0)
            result = generator.load_lazy_data(lazy_data)
            assert isinstance(result, CompletionAPIData)
            assert result.prompt == generator.prompts[i]

    def test_single_turn_load_lazy_data_chat(self) -> None:
        """Test load_lazy_data for single-turn Chat API."""
        from inference_perf.apis.chat import ChatCompletionAPIData
        from inference_perf.apis.base import LazyLoadInferenceAPIData

        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=False,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        for i in range(len(generator.prompts)):
            lazy_data = LazyLoadInferenceAPIData(data_index=i, prefered_worker_id=0)
            result = generator.load_lazy_data(lazy_data)
            assert isinstance(result, ChatCompletionAPIData)

            shared_prefix, question = generator.prompt_pairs[i]
            assert result.messages[0].content == shared_prefix
            assert result.messages[1].content == question

    def test_single_turn_prompts_and_pairs_are_aligned(self) -> None:
        """Test that prompts and prompt_pairs remain aligned after shuffle."""
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=3,
                num_prompts_per_group=4,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=False,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        # Verify prompts and prompt_pairs are aligned (shuffled together)
        for i in range(len(generator.prompts)):
            shared_prefix, question = generator.prompt_pairs[i]
            expected_prompt = shared_prefix + " " + question
            assert generator.prompts[i] == expected_prompt, f"Mismatch at index {i}"


class TestMultiTurnChatFlow:
    """End-to-end tests simulating actual multi-turn chat flow."""

    @pytest.mark.asyncio
    async def test_chat_api_multiturn_payload_flow(self) -> None:
        """Test that Chat API multi-turn correctly builds message history."""
        from inference_perf.apis.chat import ChatMessage

        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=1,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        user_id = 0
        user_session = generator.user_sessions[user_id]
        group_id = user_session.group_id
        shared_prefix = generator.shared_prefixes_by_group[group_id]

        # Verify initial context is [system message]
        assert isinstance(user_session.context, list)
        assert len(user_session.context) == 1
        assert user_session.context[0].role == "system"
        assert user_session.context[0].content == shared_prefix

        # === Round 0 ===
        lazy_data = LazyLoadInferenceAPIData(data_index=user_id, prefered_worker_id=0)
        result_r0 = generator.load_lazy_data(lazy_data)

        # Get payload for round 0
        payload_r0 = await result_r0.to_payload("test-model", 100, False, False)

        # Should have [system, user]
        assert len(payload_r0["messages"]) == 2
        assert payload_r0["messages"][0]["role"] == "system"
        assert payload_r0["messages"][0]["content"] == shared_prefix
        assert payload_r0["messages"][1]["role"] == "user"
        question_r0 = generator.questions_by_group[group_id][0]
        assert payload_r0["messages"][1]["content"] == question_r0

        # Simulate response and update context
        result_r0.model_response = "Assistant response for round 0"
        # Manually update context as process_response would
        new_history = list(result_r0._session_context)
        new_messages = [msg for msg in result_r0.messages if msg.role != "system"]
        new_history.extend(new_messages)
        new_history.append(ChatMessage(role="assistant", content=result_r0.model_response))
        # update_context also releases the lock acquired in get_context
        user_session.update_context(new_history)

        # Verify context after round 0: [system, user, assistant]
        assert len(user_session.context) == 3
        assert user_session.context[0].role == "system"
        assert user_session.context[1].role == "user"
        assert user_session.context[2].role == "assistant"

        # === Round 1 ===
        num_users = len(generator.user_sessions)
        data_index_r1 = num_users + user_id  # Round 1, same user
        lazy_data_r1 = LazyLoadInferenceAPIData(data_index=data_index_r1, prefered_worker_id=0)
        result_r1 = generator.load_lazy_data(lazy_data_r1)

        # Get payload for round 1
        payload_r1 = await result_r1.to_payload("test-model", 100, False, False)

        # Should have [system, user_r0, assistant_r0, user_r1]
        assert len(payload_r1["messages"]) == 4
        assert payload_r1["messages"][0]["role"] == "system"
        assert payload_r1["messages"][1]["role"] == "user"
        assert payload_r1["messages"][2]["role"] == "assistant"
        assert payload_r1["messages"][3]["role"] == "user"

        # Verify round 1 uses different question from same group
        question_r1 = generator.questions_by_group[group_id][1]
        assert payload_r1["messages"][3]["content"] == question_r1
        assert question_r0 != question_r1  # Different questions

        # System message should be the same across rounds
        assert payload_r1["messages"][0]["content"] == shared_prefix

    @pytest.mark.asyncio
    async def test_completion_api_multiturn_payload_flow(self) -> None:
        """Test that Completion API multi-turn correctly builds prompts."""
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=1,
                num_prompts_per_group=2,
                system_prompt_len=10,
                question_len=5,
                output_len=10,
                enable_multi_turn_chat=True,
            ),
        )
        tokenizer = create_mock_tokenizer()

        generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

        user_id = 0
        user_session = generator.user_sessions[user_id]
        group_id = user_session.group_id
        shared_prefix = generator.shared_prefixes_by_group[group_id]

        # Verify initial context is shared_prefix string
        assert isinstance(user_session.context, str)
        assert user_session.context == shared_prefix

        # === Round 0 ===
        lazy_data = LazyLoadInferenceAPIData(data_index=user_id, prefered_worker_id=0)
        result_r0 = generator.load_lazy_data(lazy_data)

        # The prompt should be only the question (not shared_prefix + question)
        question_r0 = generator.questions_by_group[group_id][0]
        assert result_r0.prompt == question_r0

        # Get payload for round 0
        payload_r0 = await result_r0.to_payload("test-model", 100, False, False)

        # The payload prompt should be: context + " " + question = shared_prefix + " " + question
        expected_prompt_r0 = shared_prefix + " " + question_r0
        assert payload_r0["prompt"] == expected_prompt_r0

        # Simulate response and update context
        result_r0.model_response = "Response 0"
        new_context = result_r0.prompt + " " + result_r0.model_response
        # update_context also releases the lock acquired in get_context
        user_session.update_context(new_context)

        # === Round 1 ===
        num_users = len(generator.user_sessions)
        data_index_r1 = num_users + user_id
        lazy_data_r1 = LazyLoadInferenceAPIData(data_index=data_index_r1, prefered_worker_id=0)
        result_r1 = generator.load_lazy_data(lazy_data_r1)

        question_r1 = generator.questions_by_group[group_id][1]
        assert result_r1.prompt == question_r1

        # Get payload for round 1
        payload_r1 = await result_r1.to_payload("test-model", 100, False, False)

        # The payload prompt should include full history + new question
        # context is now: "shared_prefix question_r0 Response 0"
        # new prompt: context + " " + question_r1
        expected_prompt_r1 = new_context + " " + question_r1
        assert payload_r1["prompt"] == expected_prompt_r1
