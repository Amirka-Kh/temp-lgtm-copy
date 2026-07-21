from typing import cast
from unittest import mock

import rich.layout
import rich.markdown
import rich.panel
from lgtm_ai.ai.schemas import (
    CodeSuggestion,
    CodeSuggestionOffset,
    GuideChecklistItem,
    GuideKeyChange,
    GuideReference,
    GuideResponse,
    PublishMetadata,
    Review,
    ReviewComment,
    ReviewGuide,
    ReviewResponse,
)
from lgtm_ai.formatters.pretty import PrettyFormatter
from lgtm_ai.git_client.schemas import PRDiff
from rich.console import Group


def test_format_summary_section() -> None:
    review = Review(
        pr_diff=mock.Mock(spec=PRDiff),
        metadata=mock.MagicMock(spec=PublishMetadata),
        review_response=ReviewResponse(summary="Test summary", comments=[], raw_score=3),
    )

    formatter = PrettyFormatter()

    panel = formatter.format_review_summary_section(review)

    assert panel.title == "🦉 lgtm Review"
    assert panel.subtitle == "Оценка: Needs Work 🔧"
    assert isinstance(panel.renderable, rich.markdown.Markdown)
    assert panel.renderable.markup == "Test summary"


def test_format_comments_section() -> None:
    comments = [
        ReviewComment(
            new_path="test.py",
            old_path="test.py",
            line_number=1,
            relative_line_number=1,
            severity="LOW",
            category="Testing",
            comment="Test comment",
            quote_snippet="def test():\n    pass\n",
            is_comment_on_new_path=False,
            programming_language="Python",
        )
        for _ in range(3)
    ]

    formatter = PrettyFormatter()
    group = formatter.format_review_comments_section(comments)
    assert len(group.renderables) == 3
    assert all(isinstance(child, rich.panel.Panel) for child in group.renderables)
    assert all(cast(rich.panel.Panel, child).title == "test.py:1" for child in group.renderables)


def test_format_comments_section_no_comments() -> None:
    formatter = PrettyFormatter()
    group = formatter.format_review_comments_section([])
    assert len(group.renderables) == 0


def test_format_comment_with_snippet_and_suggestion() -> None:
    comment = ReviewComment(
        new_path="test.py",
        old_path="test.py",
        line_number=1,
        relative_line_number=1,
        severity="LOW",
        category="Testing",
        comment="Test comment",
        quote_snippet="def test():\n    pass\n",
        suggestion=CodeSuggestion(
            start_offset=CodeSuggestionOffset(offset=1, direction="-"),
            end_offset=CodeSuggestionOffset(offset=2, direction="+"),
            snippet="def test_example():\n    pass\n",
            programming_language="Python",
            ready_for_replacement=True,
        ),
        is_comment_on_new_path=False,
        programming_language="Python",
    )

    panel = PrettyFormatter().format_review_comment(comment)

    group = cast(Group, panel.renderable)
    snippet_panel = cast(rich.panel.Panel, group.renderables[0])
    suggestion_panel = cast(rich.panel.Panel, group.renderables[4])
    assert snippet_panel.title == "Фрагмент кода"
    assert suggestion_panel.title == "Предложение кода"


def test_format_guide() -> None:
    guide = ReviewGuide(
        pr_diff=mock.Mock(spec=PRDiff),
        guide_response=GuideResponse(
            summary="Guide summary",
            key_changes=[GuideKeyChange(file_name="test.py", description="Key change")],
            checklist=[GuideChecklistItem(description="Checklist item")],
            references=[GuideReference(title="Reference", url="https://example.com")],
        ),
        metadata=mock.MagicMock(spec=PublishMetadata),
    )

    layout = PrettyFormatter().format_guide(guide)

    assert isinstance(layout, rich.layout.Layout)
