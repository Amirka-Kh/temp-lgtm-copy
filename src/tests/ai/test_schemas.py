"""Tests for LLM output parsing helpers in ai/schemas.py.

Covers escape normalization, bracket matching with string-aware extraction,
and embedded comment JSON parsing — including the real-world edge case where
LLM output contains inner brackets inside JSON string values (e.g. result["text"]
in a quote_snippet).
"""

import json

from lgtm_ai.ai.schemas import (
    ReviewComment,
    ReviewResponse,
    _extract_embedded_comments_from_summary,
    _normalize_llm_escapes,
    _parse_comments_json,
    _try_extract_inline_json_array,
)

# ── _normalize_llm_escapes ─────────────────────────────────────────────────


class TestNormalizeLlmEscapes:
    def test_no_change_when_clean(self) -> None:
        assert _normalize_llm_escapes("hello world") == "hello world"

    def test_literal_backslash_n_to_newline(self) -> None:
        assert _normalize_llm_escapes("line1\\nline2") == "line1\nline2"

    def test_literal_backslash_t_to_tab(self) -> None:
        assert _normalize_llm_escapes("col1\\tcol2") == "col1\tcol2"

    def test_both_escapes(self) -> None:
        result = _normalize_llm_escapes("a\\nb\\tc")
        # \\n → newline, \\t → tab
        assert result == "a\nb\tc"

    def test_xml_tags_stripped(self) -> None:
        result = _normalize_llm_escapes("<parameter=summary>hello</parameter>")
        assert result == "hello"

    def test_mixed_escapes_and_tags(self) -> None:
        result = _normalize_llm_escapes("<parameter=x>\\nval\\ntext</parameter>")
        assert result == "\nval\ntext"

    def test_none_input(self) -> None:
        assert _normalize_llm_escapes(None) is None  # type: ignore[arg-type]


# ── _try_extract_inline_json_array ─────────────────────────────────────────


class TestTryExtractInlineJsonArray:
    """Tests for bracket-matching extraction that respects string boundaries."""

    def test_simple_array_after_text(self) -> None:
        text = 'PR looks good, [ { "old_path": "a.py", "new_path": "a.py", "comment": "fix this", "category": "Quality", "severity": "LOW", "line_number": 1, "relative_line_number": 0, "is_comment_on_new_path": true, "programming_language": "Python" } ]'
        result = _try_extract_inline_json_array(text)
        assert result is not None
        clean_summary, json_str = result
        assert clean_summary == "PR looks good,"
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["old_path"] == "a.py"

    def test_inner_brackets_in_string_value(self) -> None:
        """The bug we fixed: result["text"] inside a quote_snippet.

        The previous implementation used rfind('[') which found the inner [
        in result["text"] instead of the array-opening bracket, producing
        invalid JSON like 'text"], ...'.
        """
        # Simulate real LLM output with nested brackets in string values
        text = (
            'PR вносит изменения..., "comments": [ { "old_path": "a.py", "new_path": "a.py", '
            '"comment": "test", "category": "Quality", "severity": "LOW", '
            '"line_number": 1, "relative_line_number": 0, "is_comment_on_new_path": true, '
            '"programming_language": "Python", '
            '"quote_snippet": "result[\'text\']" }, { "old_path": "b.py", "new_path": "b.py", '
            '"comment": "test2", "category": "Quality", "severity": "HIGH", '
            '"line_number": 5, "relative_line_number": 3, "is_comment_on_new_path": true, '
            '"programming_language": "Python" } ], "raw_score": 3 }'
        )
        result = _try_extract_inline_json_array(text)
        assert result is not None
        clean_summary, json_str = result
        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert parsed[0]["old_path"] == "a.py"
        assert parsed[1]["old_path"] == "b.py"

    def test_no_brackets(self) -> None:
        assert _try_extract_inline_json_array("no brackets here") is None

    def test_empty_text(self) -> None:
        assert _try_extract_inline_json_array("") is None

    def test_unmatched_bracket(self) -> None:
        text = 'text [ { "old_path": "a.py" }'
        result = _try_extract_inline_json_array(text)
        assert result is None

    def test_multiple_arrays_only_last_valid(self) -> None:
        """When there are multiple arrays, prefer the one with comment fields."""
        text = '[ 1, 2, 3 ] more text [ { "old_path": "x", "new_path": "y", "comment": "c", "category": "Q", "severity": "L", "line_number": 1, "relative_line_number": 0, "is_comment_on_new_path": true, "programming_language": "Py" } ]'
        result = _try_extract_inline_json_array(text)
        assert result is not None
        _, json_str = result
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["old_path"] == "x"

    def test_array_followed_by_object(self) -> None:
        """Array of objects (comments) should be preferred over other patterns."""
        text = 'some text [ { "data": "value" } ] end'
        result = _try_extract_inline_json_array(text)
        # Should find the array but it has no comment fields → None
        assert result is None

    def test_nested_arrays_in_string(self) -> None:
        """Brackets inside string values should not interfere with matching."""
        text = (
            'text [ { "old_path": "a.py", "new_path": "a.py", '
            '"comment": "c", "category": "Q", "severity": "L", '
            '"line_number": 1, "relative_line_number": 0, '
            '"is_comment_on_new_path": true, "programming_language": "Py", '
            '"quote_snippet": "arr = [1, 2, 3]" } ]'
        )
        result = _try_extract_inline_json_array(text)
        assert result is not None
        _, json_str = result
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        # Verify the quote_snippet was preserved correctly
        assert parsed[0]["quote_snippet"] == "arr = [1, 2, 3]"

    def test_escape_sequences_in_json(self) -> None:
        """Handle escaped characters inside JSON strings."""
        text = (
            'text [ { "old_path": "a.py", "new_path": "a.py", '
            '"comment": "line with \\\\escaped\\\\ backslash", '
            '"category": "Q", "severity": "L", '
            '"line_number": 1, "relative_line_number": 0, '
            '"is_comment_on_new_path": true, "programming_language": "Py" } ]'
        )
        result = _try_extract_inline_json_array(text)
        assert result is not None
        _, json_str = result
        parsed = json.loads(json_str)
        assert "\\\\" in parsed[0]["comment"] or "\\" in parsed[0]["comment"]

    def test_no_comment_fields(self) -> None:
        """Array without old_path/new_path should return None."""
        text = '[ { "name": "test" }, { "value": 42 } ]'
        result = _try_extract_inline_json_array(text)
        assert result is None


# ── _extract_embedded_comments_from_summary ────────────────────────────────


class TestExtractEmbeddedCommentsFromSummary:
    """Tests for three-pattern embedded comment extraction."""

    def test_pattern1_xml_parameter(self) -> None:
        """Pattern 1: <parameter=comments> XML-like tag."""
        summary = 'Some review text<parameter=comments>[{"old_path":"a.py","new_path":"a.py","comment":"fix","category":"Quality","severity":"LOW","line_number":1,"relative_line_number":0,"is_comment_on_new_path":true,"programming_language":"Python"}]</parameter>'
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert clean == "Some review text"
        assert json_str is not None
        parsed = json.loads(json_str)
        assert len(parsed) == 1

    def test_pattern2_keyword_separator(self) -> None:
        """Pattern 2: keyword like 'or comments:' followed by newline + JSON."""
        summary = 'PR has issues\ncomments:\n[{"old_path":"a.py","new_path":"a.py","comment":"fix","category":"Quality","severity":"LOW","line_number":1,"relative_line_number":0,"is_comment_on_new_path":true,"programming_language":"Python"}]'
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert "comments:" not in clean
        assert json_str is not None

    def test_pattern3_inline_array(self) -> None:
        """Pattern 3: inline JSON array with no keyword separator."""
        summary = 'PR looks good [ { "old_path": "a.py", "new_path": "a.py", "comment": "fix", "category": "Quality", "severity": "LOW", "line_number": 1, "relative_line_number": 0, "is_comment_on_new_path": true, "programming_language": "Python" } ]'
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert "comments" not in clean.lower() or "," not in clean
        assert json_str is not None

    def test_no_pattern_matches(self) -> None:
        """Clean summary with no embedded structure."""
        summary = "The PR looks good overall."
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert clean == summary
        assert json_str is None

    def test_real_world_llm_output_with_inner_brackets(self) -> None:
        """Full integration test matching the actual LLM output that failed."""
        # This is the exact structure from the bug report
        summary = (
            'PR вносит изменения..., "comments": [ { "old_path": "scripts/parsing/pdf_to_json.py", '
            '"new_path": "scripts/parsing/pdf_to_json.py", '
            '"comment": "Критическая уязвимость безопасности: хардкод API-ключа", '
            '"category": "Security", "severity": "HIGH", "line_number": 43, '
            '"relative_line_number": 30, "is_comment_on_new_path": true, '
            '"programming_language": "Python", '
            "\"quote_snippet\": \"    api_key = os.getenv('OPENAI_API_KEY', 'sk-big-secret')\", "
            '"suggestion": null }, { "old_path": "scripts/parsing/pdf_to_json.py", '
            '"new_path": "scripts/parsing/pdf_to_json.py", '
            '"comment": "В файле есть неиспользуемые импорты", '
            '"category": "Quality", "severity": "LOW", "line_number": 9, '
            '"relative_line_number": 4, "is_comment_on_new_path": true, '
            '"programming_language": "Python", '
            '"quote_snippet": "import json\\nimport markdown", '
            '"suggestion": null }, { "old_path": "scripts/parsing/pdf_to_json.py", '
            '"new_path": "scripts/parsing/pdf_to_json.py", '
            '"comment": "Код парсинга JSON закомментирован", '
            '"category": "Quality", "severity": "LOW", "line_number": 75, '
            '"relative_line_number": 38, "is_comment_on_new_path": true, '
            '"programming_language": "Python", '
            '"quote_snippet": "    # result = json.loads(raw_json)", '
            '"suggestion": null } ], "raw_score": 3 }'
        )
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert json_str is not None, "Should extract embedded comments"
        parsed = json.loads(json_str)
        assert len(parsed) == 3
        assert parsed[0]["severity"] == "HIGH"
        assert parsed[1]["severity"] == "LOW"
        assert parsed[2]["severity"] == "LOW"


# ── _parse_comments_json ───────────────────────────────────────────────────


class TestParseCommentsJson:
    def test_top_level_list(self) -> None:
        json_str = '[{"old_path":"a.py","new_path":"a.py","comment":"c","category":"Q","severity":"L","line_number":1,"relative_line_number":0,"is_comment_on_new_path":true,"programming_language":"Py"}]'
        result = _parse_comments_json(json_str)
        assert len(result) == 1

    def test_dict_with_comments_key(self) -> None:
        json_str = '{"comments":[{"old_path":"a.py","new_path":"a.py","comment":"c","category":"Q","severity":"L","line_number":1,"relative_line_number":0,"is_comment_on_new_path":true,"programming_language":"Py"}]}'
        result = _parse_comments_json(json_str)
        assert len(result) == 1

    def test_none_input(self) -> None:
        assert _parse_comments_json(None) == []

    def test_empty_string(self) -> None:
        assert _parse_comments_json("") == []

    def test_invalid_json(self) -> None:
        assert _parse_comments_json("{invalid json}") == []

    def test_non_list_comments_key(self) -> None:
        json_str = '{"comments": "not a list"}'
        assert _parse_comments_json(json_str) == []


# ── ReviewResponse integration ─────────────────────────────────────────────


class TestReviewResponseEmbeddedComments:
    """Integration tests for ReviewResponse's @model_validator."""

    def test_review_response_with_embedded_array(self) -> None:
        """When LLM puts comments array inside summary, it should be extracted."""
        # Minimal embedded structure
        summary = 'Issues found [ { "old_path": "a.py", "new_path": "a.py", "comment": "fix", "category": "Quality", "severity": "LOW", "line_number": 1, "relative_line_number": 0, "is_comment_on_new_path": true, "programming_language": "Python" } ]'
        response = ReviewResponse(summary=summary, comments=[], raw_score=3)
        assert len(response.comments) == 1
        assert response.comments[0].old_path == "a.py"

    def test_review_response_normal_comments_unchanged(self) -> None:
        """When comments are properly in the comments field, extraction is skipped."""
        comment = ReviewComment(
            old_path="a.py",
            new_path="a.py",
            comment="fix",
            category="Quality",
            severity="LOW",
            line_number=1,
            relative_line_number=0,
            is_comment_on_new_path=True,
            programming_language="Python",
        )
        response = ReviewResponse(
            summary="PR has issues",
            comments=[comment],
            raw_score=3,
        )
        assert len(response.comments) == 1
        assert response.summary == "PR has issues"

    def test_review_response_no_extraction_when_comments_exist(self) -> None:
        """If comments are already populated, embedded extraction is skipped."""
        comment = ReviewComment(
            old_path="a.py",
            new_path="a.py",
            comment="existing",
            category="Correctness",
            severity="HIGH",
            line_number=5,
            relative_line_number=2,
            is_comment_on_new_path=True,
            programming_language="Python",
        )
        # Summary also contains embedded array — but extraction should be skipped
        summary = 'Issues [ { "old_path": "b.py", "new_path": "b.py", "comment": "embedded", "category": "Q", "severity": "L", "line_number": 1, "relative_line_number": 0, "is_comment_on_new_path": true, "programming_language": "Py" } ]'
        response = ReviewResponse(summary=summary, comments=[comment], raw_score=3)
        assert len(response.comments) == 1
        assert response.comments[0].comment == "existing"


class TestExtractEmbeddedCommentsRawScoreStripping:
    """Tests for trailing raw_score artifact removal after JSON array extraction."""

    def test_raw_score_stripped_after_array(self) -> None:
        """When LLM leaks 'raw_score': N after the comments array, it should be removed."""
        summary = (
            'PR has some issues [ { "old_path": "a.py", "new_path": "a.py", '
            '"comment": "fix this", "category": "Quality", "severity": "LOW", '
            '"line_number": 1, "relative_line_number": 0, '
            '"is_comment_on_new_path": true, "programming_language": "Python" } ], "raw_score": 2'
        )
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert json_str is not None
        assert "raw_score" not in clean
        assert clean == "PR has some issues"

    def test_raw_score_stripped_with_space_before_bracket(self) -> None:
        """Pattern where whitespace separates summary from [ and raw_score follows ]."""
        summary = (
            'Overall the PR is okay [ { "old_path": "x.py", "new_path": "x.py", '
            '"comment": "minor fix", "category": "Quality", "severity": "LOW", '
            '"line_number": 1, "relative_line_number": 0, '
            '"is_comment_on_new_path": true, "programming_language": "Py" } ] "raw_score": 4'
        )
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert json_str is not None
        assert "raw_score" not in clean

    def test_raw_score_with_quoted_number(self) -> None:
        """Raw score with quoted number like '"3"' should also be stripped."""
        summary = (
            'Some text [ { "old_path": "a.py", "new_path": "a.py", '
            '"comment": "c", "category": "Q", "severity": "L", '
            '"line_number": 1, "relative_line_number": 0, '
            '"is_comment_on_new_path": true, "programming_language": "Py" } ], "raw_score": "3"'
        )
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert json_str is not None
        assert "raw_score" not in clean

    def test_no_false_positive_on_clean_summary(self) -> None:
        """A summary that legitimately mentions 'score' should not be mangled."""
        summary = "The PR looks good overall."
        clean, json_str = _extract_embedded_comments_from_summary(summary)
        assert clean == summary
        assert json_str is None


class TestReviewResponseEscapeNormalization:
    """Tests that ReviewResponse normalizes escape sequences in summary field."""

    def test_summary_escapes_normalized_by_before_validator(self) -> None:
        r"""BeforeValidator on summary should convert literal \\n to real newlines."""
        summary = "Line 1\\nLine 2"
        response = ReviewResponse(summary=summary, comments=[], raw_score=3)
        assert "\\n" not in response.summary
        assert "\n" in response.summary

    def test_summary_xml_tags_stripped_by_before_validator(self) -> None:
        """BeforeValidator should strip structural XML-like tags from summary."""
        summary = "<parameter=content>PR looks good</parameter>"
        response = ReviewResponse(summary=summary, comments=[], raw_score=3)
        assert "<parameter" not in response.summary
        assert "PR looks good" in response.summary

    def test_full_integration_escaped_and_embedded(self) -> None:
        r"""End-to-end: escaped sequences + embedded comments + raw_score artifact.

        This mirrors the real-world bug where the LLM output contained:
        - Literal \\n escape sequences in summary text
        - XML-like tags (<parameter=content>)
        - Embedded comments array after ]
        - Trailing "raw_score": N after the array
        """
        summary = (
            "<parameter=content>PR вносит изменения\\n\\nВ целом хорошо</parameter> "
            '[ { "old_path": "a.py", "new_path": "a.py", '
            '"comment": "fix", "category": "Quality", "severity": "LOW", '
            '"line_number": 1, "relative_line_number": 0, '
            '"is_comment_on_new_path": true, "programming_language": "Python" } ], "raw_score": 2'
        )
        response = ReviewResponse(summary=summary, comments=[], raw_score=3)
        # Comments extracted from embedded array
        assert len(response.comments) == 1
        # Summary is clean: no XML tags, no escaped newlines, no raw_score artifact
        assert "<parameter" not in response.summary
        assert "raw_score" not in response.summary
        assert "\\n" not in response.summary
        # The Russian text should be present with real newlines
        assert "В целом хорошо" in response.summary
