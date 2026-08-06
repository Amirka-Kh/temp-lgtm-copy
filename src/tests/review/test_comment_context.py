import pytest
from lgtm_ai.git_client.schemas import ExistingReviewComment
from lgtm_ai.review.comment_context import (
    build_existing_comments_context,
    condense_comment_body,
)

CHANGED_FILES = ["a.py", "b.py"]

LGTM_COMMENT = """#### 🦉 🎯 Correctness

> **Серьезность:** MEDIUM 🟡

```python
result = compute(x)
```

The retry counter is never reset, so the second failure aborts immediately.

`````suggestion
result = compute(x, retries=3)
`````

<details><summary>Подробности комментария</summary>

- **Файл**: `a.py`
- **Строка**: `12`
</details>
"""


def _comment(
    body: str = "a point",
    *,
    file_path: str | None = "a.py",
    line_number: int | None = 1,
    author: str = "dev",
) -> ExistingReviewComment:
    return ExistingReviewComment(body=body, author=author, file_path=file_path, line_number=line_number)


class TestCondenseCommentBody:
    def test_strips_details_blocks(self) -> None:
        condensed = condense_comment_body(LGTM_COMMENT)

        assert "<details>" not in condensed
        assert "Подробности комментария" not in condensed

    def test_strips_fenced_code_and_suggestion_blocks(self) -> None:
        condensed = condense_comment_body(LGTM_COMMENT)

        assert "result = compute(x)" not in condensed
        assert "result = compute(x, retries=3)" not in condensed
        assert "[code omitted]" in condensed

    def test_keeps_the_point_being_made(self) -> None:
        condensed = condense_comment_body(LGTM_COMMENT)

        assert "The retry counter is never reset" in condensed

    def test_strips_html_comments(self) -> None:
        assert condense_comment_body("<!-- marker --> visible") == "visible"

    def test_condensing_shrinks_a_real_lgtm_comment(self) -> None:
        assert len(condense_comment_body(LGTM_COMMENT)) < len(LGTM_COMMENT) / 2

    def test_truncates_to_the_budget(self) -> None:
        condensed = condense_comment_body("word " * 500, max_chars=100)

        assert len(condensed) <= 100 + len(" […]")
        assert condensed.endswith(" […]")

    def test_does_not_truncate_what_already_fits(self) -> None:
        assert condense_comment_body("short", max_chars=100) == "short"


class TestBuildExistingCommentsContext:
    def test_no_comments(self) -> None:
        assert build_existing_comments_context([], changed_files=CHANGED_FILES) == []

    def test_drops_comments_on_files_outside_the_diff(self) -> None:
        comments = [_comment(file_path="a.py"), _comment(file_path="untouched.py")]

        result = build_existing_comments_context(comments, changed_files=CHANGED_FILES)

        assert [c.file_path for c in result] == ["a.py"]

    def test_keeps_pr_level_comments(self) -> None:
        result = build_existing_comments_context(
            [_comment(file_path=None, line_number=None)], changed_files=CHANGED_FILES
        )

        assert len(result) == 1

    def test_drops_comments_that_are_empty_once_condensed(self) -> None:
        result = build_existing_comments_context([_comment(body="```\njust code\n```")], changed_files=CHANGED_FILES)

        assert result == []

    def test_respects_the_comment_count_cap(self) -> None:
        comments = [_comment(body=f"point {i}") for i in range(100)]

        result = build_existing_comments_context(comments, changed_files=CHANGED_FILES, max_comments=10)

        assert len(result) == 10

    def test_respects_the_total_character_budget(self) -> None:
        comments = [_comment(body="x" * 100) for _ in range(100)]

        result = build_existing_comments_context(
            comments, changed_files=CHANGED_FILES, max_comments=100, max_total_chars=550
        )

        assert sum(len(c.body) for c in result) <= 550

    def test_a_huge_thread_stays_within_the_budget(self) -> None:
        """The whole point of the module: comment volume must not grow the prompt without bound."""
        comments = [_comment(body=LGTM_COMMENT, author="lgtm") for _ in range(1000)]

        result = build_existing_comments_context(comments, changed_files=CHANGED_FILES)

        assert sum(len(c.body) for c in result) <= 15_000
        assert len(result) <= 50

    def test_marks_lgtm_authored_comments(self) -> None:
        comments = [_comment(body=LGTM_COMMENT), _comment(body="a human point")]

        result = build_existing_comments_context(comments, changed_files=CHANGED_FILES)

        assert [c.is_own_comment for c in result] == [True, False]

    def test_own_comments_win_the_budget_over_human_ones(self) -> None:
        """Comments lgtm wrote on an earlier run are the ones it would write again, so they must survive the cap."""
        own = [_comment(body=f"🦉 own point {i}") for i in range(5)]
        human = [_comment(body=f"human point {i}") for i in range(20)]

        result = build_existing_comments_context(own + human, changed_files=CHANGED_FILES, max_comments=5)

        assert all(c.is_own_comment for c in result)

    def test_preserves_chronological_order(self) -> None:
        comments = [_comment(body="🦉 first"), _comment(body="second human"), _comment(body="🦉 third")]

        result = build_existing_comments_context(comments, changed_files=CHANGED_FILES)

        assert [c.body for c in result] == ["🦉 first", "second human", "🦉 third"]

    @pytest.mark.parametrize("field", ["author", "file_path", "line_number", "is_resolved"])
    def test_preserves_comment_metadata(self, field: str) -> None:
        comment = ExistingReviewComment(
            body="a point", author="reviewer", file_path="a.py", line_number=42, is_resolved=True
        )

        result = build_existing_comments_context([comment], changed_files=CHANGED_FILES)

        assert getattr(result[0], field) == getattr(comment, field)
