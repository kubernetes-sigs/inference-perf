import asyncio
import unittest
from unittest.mock import MagicMock
from typing import Any

from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    SharedPrefix,
)
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.datagen.base import LazyLoadDataMixin
from inference_perf.apis import LazyLoadInferenceAPIData
from inference_perf.apis.user_session import LocalUserSession


def _make_mock_tokenizer(vocab_size: int = 1000) -> MagicMock:
    mock_tokenizer = MagicMock()
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf_tok.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
    mock_tokenizer.get_tokenizer.return_value = hf_tok
    return mock_tokenizer


class TestMultiTurnHang(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        try:
            LocalUserSession.clear_instances()
        except AttributeError:
            pass

    async def test_worker_simulation_no_hang(self) -> None:
        sp = SharedPrefix(
            num_groups=1,
            num_prompts_per_group=5,
            enable_multi_turn_chat=True,
            seed=42,
            system_prompt_len=10,
            question_len=10,
            output_len=10,
        )
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=sp)

        datagen = SharedPrefixDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # Simulate Worker 0 processing requests for session 0
        # Request 0 (round 0)
        lazy_data_0 = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        data_0 = LazyLoadDataMixin.get_request(datagen, lazy_data_0)

        # Request 1 (round 1) - Use data_index=5 to map to same user_id (0)
        lazy_data_1 = LazyLoadInferenceAPIData(data_index=5, preferred_worker_id=0)
        data_1 = LazyLoadDataMixin.get_request(datagen, lazy_data_1)

        # Simulate IPC serialization/deserialization (Pydantic copying behavior)
        import pickle

        data_0 = pickle.loads(pickle.dumps(data_0))
        data_1 = pickle.loads(pickle.dumps(data_1))

        # Assert instances are identical (Registry working)
        self.assertIs(data_0.user_session, data_1.user_session)

        await data_0.to_payload("model", 10, True, False)
        print("Processed request 0 payload (Lock acquired)")

        # Simulate successful response processing for request 0
        mock_response = MagicMock()
        mock_json_fut: asyncio.Future[Any] = asyncio.Future()
        mock_json_fut.set_result({"choices": [{"text": "bot response"}]})
        mock_response.json = MagicMock(return_value=mock_json_fut)

        await data_0.process_response(mock_response, api_config, datagen.tokenizer)
        print("Processed request 0 response (Lock released)")

        # This should NOT hang now!
        try:
            await asyncio.wait_for(data_1.to_payload("model", 10, True, False), timeout=2.0)
            print("Processed request 1 payload successfully (No hang!)")
        except asyncio.TimeoutError:
            self.fail("Should NOT have hung on request 1 after release!")

    async def test_worker_simulation_failure_no_hang(self) -> None:
        sp = SharedPrefix(
            num_groups=1,
            num_prompts_per_group=5,
            enable_multi_turn_chat=True,
            seed=42,
            system_prompt_len=10,
            question_len=10,
            output_len=10,
        )
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=sp)

        datagen = SharedPrefixDataGenerator(api_config, data_config, _make_mock_tokenizer())

        lazy_data_0 = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        data_0 = LazyLoadDataMixin.get_request(datagen, lazy_data_0)

        lazy_data_1 = LazyLoadInferenceAPIData(data_index=5, preferred_worker_id=0)
        data_1 = LazyLoadDataMixin.get_request(datagen, lazy_data_1)

        # Simulate IPC serialization/deserialization (Pydantic copying behavior)
        import pickle

        data_0 = pickle.loads(pickle.dumps(data_0))
        data_1 = pickle.loads(pickle.dumps(data_1))

        # Assert instances are identical (Registry working)
        self.assertIs(data_0.user_session, data_1.user_session)

        await data_0.to_payload("model", 10, True, False)

        # Simulate failure processing for request 0
        await data_0.process_failure(None, api_config, datagen.tokenizer, Exception("Simulated failure"))
        print("Processed request 0 failure (Lock released)")

        # This should NOT hang now!
        try:
            await asyncio.wait_for(data_1.to_payload("model", 10, True, False), timeout=2.0)
            print("Processed request 1 payload successfully after failure (No hang!)")
        except asyncio.TimeoutError:
            self.fail("Should NOT have hung on request 1 after failure release!")

    async def test_update_context_idempotent_on_double_release(self) -> None:
        """Regression: process_response can release the session lock and then
        raise (e.g. broken SSE stream after partial parse), causing the outer
        openai_client handler to invoke process_failure, which calls
        update_context again. The second release must be a no-op rather than
        raising RuntimeError("Lock is not acquired."), since that would kill
        the loadgen task and stall the conversation slot.
        """
        session = LocalUserSession("conv_double_release")
        await session.get_context(0)  # acquire (to_payload)
        session.update_context("first response")  # first release (process_response success)

        # Second release simulates process_failure being invoked after
        # process_response already released the lock.
        try:
            session.update_context("failure context")
        except RuntimeError as e:
            self.fail(f"Second update_context raised: {e}")

        # Lock should be free; next get_context must not hang.
        try:
            ctx = await asyncio.wait_for(session.get_context(1), timeout=2.0)
        except asyncio.TimeoutError:
            self.fail("get_context hung after double release")
        self.assertEqual(ctx, "failure context")


if __name__ == "__main__":
    unittest.main()
