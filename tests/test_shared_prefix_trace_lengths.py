import json
from pathlib import Path
from inference_perf.utils.shared_prefix_trace_reader import SharedPrefixTraceReader


def test_shared_prefix_trace_reader_lengths(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace_lengths.jsonl"
    entries = [
        {
            "timestamp": 10.0,
            "shared_prefix_length": 50,
            "tail_input_length": 25,
            "output_length": 100,
            "shared_prefix_id": 1,
        },
        {
            "timestamp": 12.5,
            "shared_prefix_length": 50,
            "tail_input_length": 30,
            "output_length": 100,
            "shared_prefix_id": 1,
        },
        {
            "timestamp": 15.0,
            "shared_prefix_length": 40,
            "tail_input_length": 20,
            "output_length": 50,
            "shared_prefix_id": 2,
        },
    ]

    with open(trace_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    reader = SharedPrefixTraceReader()
    traces = reader.load_entries(trace_file)

    assert len(traces) == 3
    # First timestamp should be normalized to 0.0
    assert traces[0].timestamp == 0.0
    # Second timestamp: 12.5 - 10.0 = 2.5
    assert traces[1].timestamp == 2.5
    # Third timestamp: 15.0 - 10.0 = 5.0
    assert traces[2].timestamp == 5.0

    # Token counts
    assert traces[0].shared_prefix_length == 50
    assert traces[0].tail_input_length == 25
    assert traces[1].shared_prefix_length == 50
    assert traces[1].tail_input_length == 30
    assert traces[2].shared_prefix_length == 40
    assert traces[2].tail_input_length == 20

    # Output lengths
    assert traces[0].output_length == 100
    assert traces[1].output_length == 100
    assert traces[2].output_length == 50

    assert traces[1].shared_prefix_id == 1
    assert traces[2].shared_prefix_id == 2
