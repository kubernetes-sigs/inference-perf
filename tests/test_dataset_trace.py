import tempfile
import traceback
from pathlib import Path
from inference_perf.apis import LazyLoadInferenceAPIData, CompletionAPIData, ChatCompletionAPIData
from inference_perf.datagen.base import LazyLoadDataMixin
from inference_perf.datagen.dataset_trace_datagen import DatasetTraceDataGenerator
from inference_perf.utils.trace_reader import DatasetTraceReader, DatasetTraceEntry
from inference_perf.config import APIConfig, DataConfig, APIType, TraceFormat, TraceConfig, DataGenType


def test_dataset_trace_reader_basic():
    """Test basic JSONL parsing with text_input and output_length."""
    content = """{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing in simple terms.", "output_length": 100}
{"text_input": "Write a Python function for fibonacci.", "output_length": 150}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
        assert entries[0].text_input == "What is the capital of France?"
        assert entries[0].output_length == 20
        assert entries[1].text_input == "Explain quantum computing in simple terms."
        assert entries[1].output_length == 100
        assert entries[2].text_input == "Write a Python function for fibonacci."
        assert entries[2].output_length == 150
        print("PASSED: test_dataset_trace_reader_basic")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_without_output_length():
    """Test JSONL parsing when output_length is omitted."""
    content = """{"text_input": "What is the capital of France?"}
{"text_input": "Another prompt without output length"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0].text_input == "What is the capital of France?"
        assert entries[0].output_length is None
        assert entries[1].text_input == "Another prompt without output length"
        assert entries[1].output_length is None
        print("PASSED: test_dataset_trace_reader_without_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_mixed():
    """Test JSONL parsing with mixed entries (some with output_length, some without)."""
    content = """{"text_input": "Prompt with length", "output_length": 50}
{"text_input": "Prompt without length"}
{"text_input": "Another with length", "output_length": 200}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
        assert entries[0].output_length == 50
        assert entries[1].output_length is None
        assert entries[2].output_length == 200
        print("PASSED: test_dataset_trace_reader_mixed")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_stream_entries():
    """Test streaming entries from JSONL file."""
    content = """{"text_input": "First prompt", "output_length": 10}
{"text_input": "Second prompt", "output_length": 20}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = list(reader.stream_entries(temp_path))

        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0].text_input == "First prompt"
        assert entries[0].output_length == 10
        assert entries[1].text_input == "Second prompt"
        assert entries[1].output_length == 20
        print("PASSED: test_dataset_trace_reader_stream_entries")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_skips_empty_lines():
    """Test that empty lines are skipped."""
    content = """{"text_input": "First prompt"}

{"text_input": "Second prompt"}

"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        print("PASSED: test_dataset_trace_reader_skips_empty_lines")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_handles_invalid_json():
    """Test that invalid JSON lines are skipped with a warning."""
    content = """{"text_input": "Valid prompt"}
{invalid json here}
{"text_input": "Another valid prompt"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should skip the invalid line and parse 2 valid entries
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert entries[0].text_input == "Valid prompt"
        assert entries[1].text_input == "Another valid prompt"
        print("PASSED: test_dataset_trace_reader_handles_invalid_json")
    finally:
        temp_path.unlink()


def test_dataset_trace_reader_handles_missing_text_input():
    """Test that entries missing text_input field are skipped."""
    content = """{"text_input": "Valid prompt"}
{"output_length": 50}
{"text_input": "Another valid prompt"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        reader = DatasetTraceReader()
        entries = reader.load_entries(temp_path)

        # Should skip the entry without text_input
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        print("PASSED: test_dataset_trace_reader_handles_missing_text_input")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_completion_api():
    """Test DatasetTraceDataGenerator with Completion API."""
    content = """{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing.", "output_length": 100}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(2)]
        assert all(isinstance(data, LazyLoadInferenceAPIData) for data in lazy_data_list)

        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        assert len(data_list) == 2
        assert isinstance(data_list[0], CompletionAPIData)
        assert data_list[0].prompt == "What is the capital of France?"
        assert data_list[0].max_tokens == 20
        assert isinstance(data_list[1], CompletionAPIData)
        assert data_list[1].prompt == "Explain quantum computing."
        assert data_list[1].max_tokens == 100
        print("PASSED: test_dataset_trace_datagen_completion_api")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_chat_api():
    """Test DatasetTraceDataGenerator with Chat API."""
    content = """{"text_input": "What is the capital of France?", "output_length": 20}
{"text_input": "Explain quantum computing.", "output_length": 100}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Chat)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(2)]

        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        assert len(data_list) == 2
        assert isinstance(data_list[0], ChatCompletionAPIData)
        assert len(data_list[0].messages) == 1
        assert data_list[0].messages[0].role == "user"
        assert data_list[0].messages[0].content == "What is the capital of France?"
        assert data_list[0].max_tokens == 20
        assert isinstance(data_list[1], ChatCompletionAPIData)
        assert data_list[1].messages[0].content == "Explain quantum computing."
        assert data_list[1].max_tokens == 100
        print("PASSED: test_dataset_trace_datagen_chat_api")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_without_output_length():
    """Test DatasetTraceDataGenerator when output_length is not specified."""
    content = """{"text_input": "What is the capital of France?"}
{"text_input": "Explain quantum computing."}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        lazy_data_list = [next(data_generator) for _ in range(2)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        # When output_length is not specified, max_tokens should be 0 (server default)
        assert data_list[0].max_tokens == 0
        assert data_list[1].max_tokens == 0
        print("PASSED: test_dataset_trace_datagen_without_output_length")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_cycles_entries():
    """Test that DatasetTraceDataGenerator cycles through entries."""
    content = """{"text_input": "First prompt", "output_length": 10}
{"text_input": "Second prompt", "output_length": 20}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        datagen = DatasetTraceDataGenerator(api_config=api_config, config=data_config)

        data_generator = datagen.get_data()
        # Get more entries than in the file to test cycling
        lazy_data_list = [next(data_generator) for _ in range(5)]
        data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_data_list]

        # Should cycle: first, second, first, second, first
        assert data_list[0].prompt == "First prompt"
        assert data_list[1].prompt == "Second prompt"
        assert data_list[2].prompt == "First prompt"
        assert data_list[3].prompt == "Second prompt"
        assert data_list[4].prompt == "First prompt"
        print("PASSED: test_dataset_trace_datagen_cycles_entries")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_requires_trace_config():
    """Test that DatasetTraceDataGenerator raises error when trace config is missing."""
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(type=DataGenType.DatasetTrace, trace=None)

    try:
        DatasetTraceDataGenerator(api_config=api_config, config=data_config)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "requires a trace config" in str(e)
        print("PASSED: test_dataset_trace_datagen_requires_trace_config")


def test_dataset_trace_datagen_requires_correct_format():
    """Test that DatasetTraceDataGenerator raises error for wrong trace format."""
    content = """{"text_input": "Test prompt"}
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        # Use wrong format
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.AZURE_PUBLIC_DATASET)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        DatasetTraceDataGenerator(api_config=api_config, config=data_config)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "DatasetTrace" in str(e)
        print("PASSED: test_dataset_trace_datagen_requires_correct_format")
    finally:
        temp_path.unlink()


def test_dataset_trace_datagen_empty_file():
    """Test that DatasetTraceDataGenerator raises error for empty file."""
    content = ""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write(content)
        f.flush()
        temp_path = Path(f.name)

    try:
        api_config = APIConfig(type=APIType.Completion)
        trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.DATASET_TRACE)
        data_config = DataConfig(type=DataGenType.DatasetTrace, trace=trace_config)

        DatasetTraceDataGenerator(api_config=api_config, config=data_config)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "No valid entries" in str(e)
        print("PASSED: test_dataset_trace_datagen_empty_file")
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    tests = [
        test_dataset_trace_reader_basic,
        test_dataset_trace_reader_without_output_length,
        test_dataset_trace_reader_mixed,
        test_dataset_trace_reader_stream_entries,
        test_dataset_trace_reader_skips_empty_lines,
        test_dataset_trace_reader_handles_invalid_json,
        test_dataset_trace_reader_handles_missing_text_input,
        test_dataset_trace_datagen_completion_api,
        test_dataset_trace_datagen_chat_api,
        test_dataset_trace_datagen_without_output_length,
        test_dataset_trace_datagen_cycles_entries,
        test_dataset_trace_datagen_requires_trace_config,
        test_dataset_trace_datagen_requires_correct_format,
        test_dataset_trace_datagen_empty_file,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")

