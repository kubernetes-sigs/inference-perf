import unittest
from unittest.mock import patch, MagicMock
from inference_perf.utils.cli_summary import (
    extract_stage_id,
    extract_session_stage_id,
    print_summary_table,
    print_session_summary_tables,
)
from inference_perf.utils.report_file import ReportFile


class TestCliSummary(unittest.TestCase):
    def test_extract_stage_id(self) -> None:
        self.assertEqual(extract_stage_id("stage_0_lifecycle_metrics"), 0)
        self.assertEqual(extract_stage_id("stage_12_lifecycle_metrics"), 12)
        self.assertIsNone(extract_stage_id("invalid_name"))
        self.assertIsNone(extract_stage_id("stage_abc_lifecycle_metrics"))

    @patch("inference_perf.utils.cli_summary.rprint")
    def test_print_summary_table_empty(self, mock_rprint: MagicMock) -> None:
        print_summary_table([])
        mock_rprint.assert_called_once_with("[yellow]No per-stage lifecycle metrics found to display summary table.[/yellow]")

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_print_summary_table_with_reports(self, mock_console_print: MagicMock) -> None:
        # Create a mock report file
        contents = {
            "load_summary": {"requested_rate": 10.0, "achieved_rate": 9.5},
            "successes": {
                "count": 95,
                "latency": {
                    "time_to_first_token": {"mean": 0.05, "median": 0.045, "p90": 0.08},
                    "inter_token_latency": {"mean": 0.02, "median": 0.018, "p90": 0.03},
                },
                "throughput": {
                    "requests_per_sec": 9.5,
                    "input_tokens_per_sec": 100.0,
                    "output_tokens_per_sec": 200.0,
                    "total_tokens_per_sec": 300.0,
                },
                "prompt_len": {"mean": 50.0, "median": 45.0, "p90": 60.0},
                "output_len": {"mean": 150.0, "median": 140.0, "p90": 180.0},
            },
            "failures": {"count": 5},
        }
        report = ReportFile(name="stage_0_lifecycle_metrics", contents=contents)
        print_summary_table([report])

        self.assertEqual(mock_console_print.call_count, 4)

    def test_extract_session_stage_id(self) -> None:
        self.assertEqual(extract_session_stage_id("stage_0_session_lifecycle_metrics"), 0)
        self.assertEqual(extract_session_stage_id("stage_3_session_lifecycle_metrics"), 3)
        self.assertIsNone(extract_session_stage_id("stage_0_lifecycle_metrics"))
        self.assertIsNone(extract_session_stage_id("invalid_name"))

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_print_session_summary_tables_skipped_when_no_session_reports(self, mock_console_print: MagicMock) -> None:
        report = ReportFile(name="stage_0_lifecycle_metrics", contents={})
        print_session_summary_tables([report])
        mock_console_print.assert_not_called()

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_print_session_summary_tables_with_data(self, mock_console_print: MagicMock) -> None:
        contents = {
            "num_sessions": 5,
            "num_sessions_succeeded": 4,
            "num_sessions_failed": 1,
            "total_events": 149,
            "total_events_completed": 80,
            "total_events_cancelled": 68,
            "sessions_per_second": 0.06,
            "session_duration_sec": {"mean": 29.08, "median": 11.97, "p90": 63.26},
            "num_events": {"mean": 29.8, "median": 28.0, "p90": 55.0},
            "total_input_tokens": {"mean": 74561.6, "median": 16510.0, "p90": 193009.2},
            "total_output_tokens": {"mean": 1184.6, "median": 444.0, "p90": 2925.2},
        }
        report = ReportFile(name="stage_0_session_lifecycle_metrics", contents=contents)
        print_session_summary_tables([report])
        # 3 session tables printed
        self.assertEqual(mock_console_print.call_count, 3)

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_print_summary_table_includes_session_tables(self, mock_console_print: MagicMock) -> None:
        request_contents = {
            "load_summary": {"requested_rate": 0.0, "achieved_rate": 1.0},
            "successes": {
                "count": 79,
                "latency": {
                    "request_latency": {"mean": 1.599, "median": 1.111, "p90": 3.321},
                    "normalized_time_per_output_token": {"mean": 0.0267, "median": 0.0228, "p90": 0.0384},
                },
                "throughput": {
                    "requests_per_sec": 0.9,
                    "input_tokens_per_sec": 4373.7,
                    "output_tokens_per_sec": 69.5,
                    "total_tokens_per_sec": 4443.2,
                },
                "prompt_len": {"mean": 4660.1, "median": 2949.0, "p90": 10938.0},
                "output_len": {"mean": 74.0, "median": 51.0, "p90": 172.3},
            },
            "failures": {"count": 1},
        }
        session_contents = {
            "num_sessions": 5,
            "num_sessions_succeeded": 4,
            "num_sessions_failed": 1,
            "total_events": 149,
            "total_events_completed": 80,
            "total_events_cancelled": 68,
            "sessions_per_second": 0.06,
            "session_duration_sec": {"mean": 29.08, "median": 11.97, "p90": 63.26},
            "num_events": {"mean": 29.8, "median": 28.0, "p90": 55.0},
            "total_input_tokens": {"mean": 74561.6, "median": 16510.0, "p90": 193009.2},
            "total_output_tokens": {"mean": 1184.6, "median": 444.0, "p90": 2925.2},
        }
        reports = [
            ReportFile(name="stage_0_lifecycle_metrics", contents=request_contents),
            ReportFile(name="stage_0_session_lifecycle_metrics", contents=session_contents),
        ]
        print_summary_table(reports)
        # 4 request tables + 3 session tables = 7
        self.assertEqual(mock_console_print.call_count, 7)


if __name__ == "__main__":
    unittest.main()
