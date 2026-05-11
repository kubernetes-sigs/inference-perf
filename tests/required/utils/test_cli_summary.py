import unittest
from unittest.mock import patch, MagicMock
from inference_perf.utils.cli_summary import extract_stage_id, print_summary_table
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


if __name__ == "__main__":
    unittest.main()
