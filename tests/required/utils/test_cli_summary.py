# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch, MagicMock
from rich.console import Console
from inference_perf.utils.cli_summary import (
    MAX_LABEL_SLICE_ROWS,
    extract_stage_id,
    is_label_slice_report,
    print_label_slice_summary_table,
    extract_session_stage_id,
    print_summary_table,
    print_session_summary_tables,
    print_error_summary_table,
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
                "prompt_tokens": {"mean": 50.0, "median": 45.0, "p90": 60.0},
                "output_len": {"mean": 150.0, "median": 140.0, "p90": 180.0},
            },
            "failures": {"count": 5},
        }
        report = ReportFile(name="stage_0_lifecycle_metrics", contents=contents)
        print_summary_table([report])

        self.assertEqual(mock_console_print.call_count, 5)

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
            "num_sessions_completed": 5,
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
        # 3 session tables printed (no TFUT data → TFUT table skipped)
        self.assertEqual(mock_console_print.call_count, 3)

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_print_session_summary_tables_with_tfut(self, mock_console_print: MagicMock) -> None:
        contents = {
            "num_sessions": 5,
            "num_sessions_completed": 5,
            "num_sessions_succeeded": 5,
            "num_sessions_failed": 0,
            "total_events": 30,
            "total_events_completed": 30,
            "total_events_cancelled": 0,
            "sessions_per_second": 0.12,
            "session_duration_sec": {"mean": 13.8, "median": 12.7, "p90": 15.7},
            "num_events": {"mean": 6.0, "median": 6.0, "p90": 6.0},
            "total_input_tokens": {"mean": 9068.0, "median": 9065.0, "p90": 9093.0},
            "total_output_tokens": {"mean": 239.0, "median": 239.0, "p90": 243.0},
            "tfut_sec": {"mean": 11.84, "median": 10.69, "p90": 13.71},
            "sessions_with_tfut": 5,
            "tfut_none_reasons": None,
        }
        report = ReportFile(name="stage_0_session_lifecycle_metrics", contents=contents)
        print_session_summary_tables([report])
        # 3 session tables + 1 TFUT table = 4
        self.assertEqual(mock_console_print.call_count, 4)

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
                "prompt_tokens": {"mean": 4660.1, "median": 2949.0, "p90": 10938.0},
                "output_len": {"mean": 74.0, "median": 51.0, "p90": 172.3},
            },
            "failures": {"count": 1},
        }
        session_contents = {
            "num_sessions": 5,
            "num_sessions_completed": 5,
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
        # 5 request tables (including error table) + 3 session tables = 8
        self.assertEqual(mock_console_print.call_count, 8)

    @patch("inference_perf.utils.cli_summary.Table.add_row")
    def test_session_error_rate_is_relative_to_completed_sessions(self, mock_add_row: MagicMock) -> None:
        """A stranded session (not yet succeeded or failed) must not dilute the error
        rate: 1 failure out of 2 completed is 50%, even though 4 sessions were dispatched."""
        contents = {
            "num_sessions": 4,
            "num_sessions_completed": 2,
            "num_sessions_succeeded": 1,
            "num_sessions_failed": 1,
            "num_sessions_not_completed": 2,
            "num_sessions_not_completed_active": 1,
            "num_sessions_not_completed_pending": 1,
            "total_events": 10,
            "total_events_completed": 5,
            "total_events_cancelled": 0,
            "sessions_per_second": 0.1,
        }
        report = ReportFile(name="stage_0_session_lifecycle_metrics", contents=contents)
        print_session_summary_tables([report])

        session_summary_row = mock_add_row.call_args_list[0].args
        assert session_summary_row[2] == "4"  # num_sessions (total)
        assert "50.0%" in session_summary_row[6]  # 1 failed / 2 completed

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_session_error_summary_table_printed_when_sessions_fail(self, mock_console_print: MagicMock) -> None:
        session_contents = {
            "num_sessions": 3,
            "num_sessions_succeeded": 1,
            "num_sessions_failed": 2,
            "failures": {
                "count": 2,
                "by_label": {
                    "predecessor_failed": {"count": 2, "messages": [{"message": "predecessor failed"}]},
                },
            },
        }
        report = ReportFile(name="stage_0_session_lifecycle_metrics", contents=session_contents)
        print_error_summary_table([report])

        printed = [call.args[0] for call in mock_console_print.call_args_list]
        titles = [t.title for t in printed]
        self.assertIn("[bold magenta]Session Error Summary[/bold magenta]", titles)

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_session_error_summary_table_omitted_when_no_failures(self, mock_console_print: MagicMock) -> None:
        """A clean replay run should not print a table whose only column is the success count."""
        session_contents = {
            "num_sessions": 3,
            "num_sessions_succeeded": 3,
            "num_sessions_failed": 0,
            "failures": {"count": 0, "by_label": {}},
        }
        report = ReportFile(name="stage_0_session_lifecycle_metrics", contents=session_contents)
        print_error_summary_table([report])

        self.assertEqual(mock_console_print.call_count, 0)


if __name__ == "__main__":
    unittest.main()


class TestSessionSummaryWithNoCompletedSessions(unittest.TestCase):
    """A stage where every session was truncated still has to print.

    ``summarize()`` returns ``None`` for an empty list, and ``session_duration_sec`` is
    summarized over completed sessions only. A duration-bounded stage whose deadline is
    shorter than one session completes none of them, so the key is present with value
    ``None`` -- not absent -- and ``contents.get(key, {})`` hands back that ``None``.
    Observed for real: a 150s->8s duration stage truncated all 4 sessions and the run
    died with ``AttributeError: 'NoneType' object has no attribute 'get'`` after the
    reports were already on disk, losing every summary table.
    """

    # Matches the shape reportgen writes when no session completed: the aggregate
    # counters are real, the distributions are None.
    NO_COMPLETIONS = {
        "num_sessions": 4,
        "num_sessions_completed": 0,
        "num_sessions_succeeded": 0,
        "num_sessions_failed": 0,
        "num_sessions_not_completed": 4,
        "num_sessions_not_completed_active": 4,
        "num_sessions_not_completed_pending": 0,
        "total_events": 46,
        "total_events_completed": 20,
        "total_events_cancelled": 26,
        "sessions_per_second": 0.5,
        "session_duration_sec": None,
        "num_events": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
    }

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_all_sessions_truncated_still_prints_the_tables(self, mock_console_print: MagicMock) -> None:
        report = ReportFile(name="stage_0_session_lifecycle_metrics", contents=dict(self.NO_COMPLETIONS))

        print_session_summary_tables([report])

        self.assertEqual(mock_console_print.call_count, 3)

    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_missing_distributions_render_as_dashes_not_zeroes(self, mock_console_print: MagicMock) -> None:
        """0.00 would read as "sessions took no time"; a dash says "nothing to measure"."""
        report = ReportFile(name="stage_0_session_lifecycle_metrics", contents=dict(self.NO_COMPLETIONS))

        print_session_summary_tables([report])

        duration_table = mock_console_print.call_args_list[1][0][0]
        cells = [list(column.cells) for column in duration_table.columns]
        self.assertEqual([c[0] for c in cells], ["0", "-", "-", "-", "-", "-", "-"])


# Kept out of TestCliSummary so the ratchet does not treat the whole class as edited.
class TestCancelledColumn(unittest.TestCase):
    # A stage report with 5 successes, no failures and a cancelled block of count 3.
    # Expects the request error table to gain a "Cancelled at Teardown" column
    # showing 3, with no error label column beside it.
    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_cancelled_at_teardown_gets_its_own_column(self, mock_console_print: MagicMock) -> None:
        contents = {
            "successes": {"count": 5},
            "failures": {"count": 0, "by_label": {}},
            "cancelled": {"count": 3},
        }
        report = ReportFile(name="stage_0_lifecycle_metrics", contents=contents)
        print_error_summary_table([report])

        table = mock_console_print.call_args_list[0].args[0]
        headers = [column.header for column in table.columns]
        self.assertEqual(headers, ["Stage", "Successes", "Cancelled at Teardown"])
        self.assertEqual(table.columns[2]._cells, ["[yellow]3[/]"])

    # A stage report with 5 successes and no cancelled block. Expects no
    # "Cancelled at Teardown" column.
    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_cancelled_column_omitted_when_nothing_cancelled(self, mock_console_print: MagicMock) -> None:
        contents = {"successes": {"count": 5}, "failures": {"count": 0, "by_label": {}}}
        report = ReportFile(name="stage_0_lifecycle_metrics", contents=contents)
        print_error_summary_table([report])

        table = mock_console_print.call_args_list[0].args[0]
        self.assertEqual([column.header for column in table.columns], ["Stage", "Successes"])


# One label-slice report's contents: `count` requests of which `failed` failed, labelled tenant=<tenant>.
def _label_slice(tenant: str, count: int, failed: int = 0) -> ReportFile:
    contents = {
        "labels": {"tenant": tenant},
        "load_summary": {"count": count},
        "successes": {
            "count": count - failed,
            "latency": {"time_to_first_token": {"mean": 0.05, "p90": 0.08}},
            "throughput": {"requests_per_sec": 2.0},
        },
        "failures": {"count": failed},
    }
    return ReportFile(name=f"labels_tenant={tenant}_lifecycle_metrics", contents=contents)


# Renders the label-slice table for the given reports and returns what was printed as plain text.
def _render_label_slices(reports: list[ReportFile], width: int = 200) -> str:
    console = Console(record=True, width=width)
    with patch("inference_perf.utils.cli_summary.Console", return_value=console):
        print_label_slice_summary_table(reports)
    return console.export_text()


class TestLabelSliceSummary(unittest.TestCase):
    # labels_<...>_lifecycle_metrics is a label slice; stage, adapter and summary report names are not.
    def test_is_label_slice_report(self) -> None:
        self.assertTrue(is_label_slice_report("labels_tenant=a_lifecycle_metrics"))
        self.assertFalse(is_label_slice_report("stage_0_lifecycle_metrics"))
        self.assertFalse(is_label_slice_report("adapter_labels_x_lifecycle_metrics"))
        self.assertFalse(is_label_slice_report("summary_lifecycle_metrics"))
        self.assertFalse(is_label_slice_report("labels_tenant=a_session_metrics"))

    # With no label-slice report among the inputs nothing is printed.
    def test_prints_nothing_without_slices(self) -> None:
        self.assertEqual(_render_label_slices([ReportFile(name="stage_0_lifecycle_metrics", contents={})]), "")

    # Tenant a (10 requests, 1 failed) renders its labels, request count, 10.0% errors and TTFT in ms.
    def test_renders_one_row_per_slice(self) -> None:
        text = _render_label_slices([_label_slice("a", 10, failed=1), _label_slice("b", 4)])
        self.assertIn("tenant=a", text)
        self.assertIn("tenant=b", text)
        self.assertIn("10.0%", text)
        self.assertIn("50.0", text)
        self.assertIn("80.0", text)

    # The Goodput % column appears only when a slice reports goodput: 75.0 for tenant a, a dash for tenant b.
    def test_goodput_column_only_when_reported(self) -> None:
        with_goodput = _label_slice("a", 4)
        with_goodput.contents["successes"]["goodput_metrics"] = {"goodput_percentage": 75.0}
        self.assertNotIn("Goodput", _render_label_slices([_label_slice("b", 4)]))
        text = _render_label_slices([with_goodput, _label_slice("b", 3)])
        self.assertIn("Goodput %", text)
        self.assertIn("75.0", text)
        self.assertIn("-", text.split("tenant=b")[1])

    # In an 80-column console a tenant value of 40 "z" wraps onto further lines: all 40 are printed, none cut off.
    def test_long_labels_wrap_instead_of_truncating(self) -> None:
        text = _render_label_slices([_label_slice("z" * 40, 1)], width=80)
        self.assertNotIn("…", text)
        self.assertEqual(text.count("z"), 40)

    # A label value that looks like rich markup, "[red]x", is printed as written instead of being styled.
    def test_label_values_are_not_parsed_as_markup(self) -> None:
        self.assertIn("tenant=[red]x", _render_label_slices([_label_slice("[red]x", 1)]))

    # Slices are ordered by request count, largest first: b (9 requests) is printed above a (3).
    def test_rows_ordered_by_request_count(self) -> None:
        text = _render_label_slices([_label_slice("a", 3), _label_slice("b", 9)])
        self.assertLess(text.index("tenant=b"), text.index("tenant=a"))

    # At exactly MAX_LABEL_SLICE_ROWS slices every row is printed and no truncation note appears.
    def test_no_truncation_at_row_limit(self) -> None:
        reports = [_label_slice(f"t{i}", 100 - i) for i in range(MAX_LABEL_SLICE_ROWS)]
        text = _render_label_slices(reports)
        self.assertIn(f"tenant=t{MAX_LABEL_SLICE_ROWS - 1}", text)
        self.assertNotIn("more label slices", text)

    # Two slices over the limit: the two smallest are left out and the note counts them.
    def test_truncates_past_row_limit(self) -> None:
        reports = [_label_slice(f"t{i}", 100 - i) for i in range(MAX_LABEL_SLICE_ROWS + 2)]
        text = _render_label_slices(reports)
        self.assertNotIn(f"tenant=t{MAX_LABEL_SLICE_ROWS}", text)
        self.assertIn("2 more label slices", text)

    # print_summary_table prints the label-slice table along with the four per-stage tables.
    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_summary_table_includes_label_slices(self, mock_console_print: MagicMock) -> None:
        stage = ReportFile(name="stage_0_lifecycle_metrics", contents={"successes": {"count": 1}, "failures": {"count": 0}})
        print_summary_table([stage])
        without_slices = mock_console_print.call_count
        mock_console_print.reset_mock()
        print_summary_table([stage, _label_slice("a", 1)])
        self.assertEqual(mock_console_print.call_count, without_slices + 1)

    # With no per-stage report the label-slice table is still printed.
    @patch("inference_perf.utils.cli_summary.rprint")
    @patch("inference_perf.utils.cli_summary.Console.print")
    def test_label_slices_printed_without_stage_reports(self, mock_console_print: MagicMock, _rprint: MagicMock) -> None:
        print_summary_table([_label_slice("a", 1)])
        self.assertEqual(mock_console_print.call_count, 1)
