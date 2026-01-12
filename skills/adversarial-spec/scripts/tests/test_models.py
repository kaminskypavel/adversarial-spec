"""Tests for models module."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    ModelResponse,
    CostTracker,
    detect_agreement,
    extract_spec,
    extract_tasks,
    get_critique_summary,
    generate_diff,
    load_context_files,
)


class TestModelResponse:
    def test_create_response(self):
        response = ModelResponse(
            model="gpt-4o",
            response="This is a critique.",
            agreed=False,
            spec="# Revised Spec",
        )
        assert response.model == "gpt-4o"
        assert response.agreed is False
        assert response.spec == "# Revised Spec"

    def test_response_with_error(self):
        response = ModelResponse(
            model="gpt-4o",
            response="",
            agreed=False,
            spec=None,
            error="API timeout",
        )
        assert response.error == "API timeout"

    def test_response_with_tokens(self):
        response = ModelResponse(
            model="gpt-4o",
            response="Response",
            agreed=True,
            spec="Spec",
            input_tokens=1000,
            output_tokens=500,
            cost=0.05,
        )
        assert response.input_tokens == 1000
        assert response.output_tokens == 500
        assert response.cost == 0.05


class TestCostTracker:
    def test_add_costs(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)

        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_cost > 0

    def test_tracks_by_model(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)
        tracker.add("gemini/gemini-2.0-flash", 2000, 1000)

        assert "gpt-4o" in tracker.by_model
        assert "gemini/gemini-2.0-flash" in tracker.by_model
        assert tracker.by_model["gpt-4o"]["input_tokens"] == 1000

    def test_accumulates_for_same_model(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)
        tracker.add("gpt-4o", 1000, 500)

        assert tracker.by_model["gpt-4o"]["input_tokens"] == 2000
        assert tracker.by_model["gpt-4o"]["output_tokens"] == 1000

    def test_summary_format(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)

        summary = tracker.summary()
        assert "Cost Summary" in summary
        assert "Total tokens" in summary
        assert "Total cost" in summary


class TestDetectAgreement:
    def test_detects_agree(self):
        assert detect_agreement("I agree. [AGREE]\n[SPEC]...[/SPEC]") is True

    def test_no_agree(self):
        assert detect_agreement("I have concerns about security.") is False

    def test_partial_agree_in_word(self):
        # [AGREE] must be present as marker
        assert detect_agreement("I disagree with this approach.") is False


class TestExtractSpec:
    def test_extracts_spec(self):
        response = "Critique here.\n\n[SPEC]\n# My Spec\n\nContent\n[/SPEC]"
        spec = extract_spec(response)
        assert spec == "# My Spec\n\nContent"

    def test_returns_none_without_tags(self):
        response = "Just a critique without spec tags."
        assert extract_spec(response) is None

    def test_returns_none_with_missing_end_tag(self):
        response = "[SPEC]Content without end tag"
        assert extract_spec(response) is None

    def test_handles_empty_spec(self):
        response = "[SPEC][/SPEC]"
        spec = extract_spec(response)
        assert spec == ""


class TestExtractTasks:
    def test_extracts_single_task(self):
        response = """
[TASK]
title: Implement auth
type: task
priority: high
description: Add OAuth2 authentication
acceptance_criteria:
- User can log in
- Session persists
[/TASK]
"""
        tasks = extract_tasks(response)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Implement auth"
        assert tasks[0]["type"] == "task"
        assert tasks[0]["priority"] == "high"
        assert len(tasks[0]["acceptance_criteria"]) == 2

    def test_extracts_multiple_tasks(self):
        response = """
[TASK]
title: Task 1
type: task
priority: high
description: First task
[/TASK]
[TASK]
title: Task 2
type: bug
priority: medium
description: Second task
[/TASK]
"""
        tasks = extract_tasks(response)
        assert len(tasks) == 2
        assert tasks[0]["title"] == "Task 1"
        assert tasks[1]["title"] == "Task 2"

    def test_handles_no_tasks(self):
        response = "No tasks here."
        tasks = extract_tasks(response)
        assert tasks == []


class TestGetCritiqueSummary:
    def test_extracts_critique_before_spec(self):
        response = "This is the critique.\n\n[SPEC]...[/SPEC]"
        summary = get_critique_summary(response)
        assert summary == "This is the critique."

    def test_truncates_long_critique(self):
        response = "A" * 500
        summary = get_critique_summary(response, max_length=100)
        assert len(summary) == 103  # 100 + "..."
        assert summary.endswith("...")

    def test_full_response_without_spec(self):
        response = "Just critique, no spec."
        summary = get_critique_summary(response)
        assert summary == "Just critique, no spec."


class TestGenerateDiff:
    def test_generates_diff(self):
        previous = "line1\nline2\nline3"
        current = "line1\nmodified\nline3"

        diff = generate_diff(previous, current)
        assert "-line2" in diff
        assert "+modified" in diff

    def test_no_diff_for_identical(self):
        content = "same\ncontent"
        diff = generate_diff(content, content)
        assert diff == ""


class TestLoadContextFiles:
    def test_loads_empty_list(self):
        result = load_context_files([])
        assert result == ""

    def test_loads_nonexistent_file(self):
        result = load_context_files(["/nonexistent/file.md"])
        assert "Error loading file" in result

    def test_formats_context(self, tmp_path):
        test_file = tmp_path / "context.md"
        test_file.write_text("# Context\n\nSome context.")

        result = load_context_files([str(test_file)])
        assert "Additional Context" in result
        assert "# Context" in result
