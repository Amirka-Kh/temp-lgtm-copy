"""Condense the comments already posted on a PR into a compact reminder for the reviewer agent.

The reviewer needs to know *what has already been said* so it does not raise the same point twice. It does not need
the comments themselves: a previous lgtm comment carries a metadata footer, a quoted code snippet and possibly a
whole suggestion block, none of which add anything the diff and the file contents do not already provide.

Feeding raw comment bodies in would therefore grow the prompt without bound on a PR with a long review history, so
everything here is about throwing away the parts that cannot cause a repeat, then capping what is left.
"""

import logging
import re
from typing import Final

from lgtm_ai.git_client.schemas import ExistingReviewComment

logger = logging.getLogger("lgtm.ai")

MAX_COMMENTS: Final[int] = 50
"""Most recent comments to keep. Older ones are the least likely to still be relevant to the current diff."""

MAX_COMMENT_CHARS: Final[int] = 400
"""Per-comment budget. Enough to recognize a point, far short of a full comment with snippets."""

MAX_TOTAL_CHARS: Final[int] = 15_000
"""Hard ceiling on the whole section, whatever the other limits allow. This is the backstop against a huge thread."""

_DETAILS_BLOCK = re.compile(r"<details>.*?</details>", re.DOTALL | re.IGNORECASE)
"""lgtm's own metadata/usage/config footers. Pure noise, and by far the largest part of its comments."""

_FENCED_BLOCK = re.compile(r"^([ \t]*)(`{3,}|~{3,}).*?^\1?\2[ \t]*$", re.DOTALL | re.MULTILINE)
"""Quoted snippets and suggestion blocks. The reviewer already has the diff and the file contents."""

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

_BLANK_RUN = re.compile(r"\n\s*\n\s*\n+")

_OWN_COMMENT_MARKER: Final[str] = "🦉"
"""Every comment lgtm publishes is rendered with this marker by the markdown formatter."""

_CODE_PLACEHOLDER: Final[str] = "[code omitted]"

_WORD = re.compile(r"\w{2,}")


def _has_substance(condensed_body: str) -> bool:
    """Whether a condensed body still says anything.

    A comment that was only a code block or a screenshot condenses down to placeholders, which cannot tell the
    reviewer what point was made — it would just spend budget.
    """
    return bool(_WORD.search(condensed_body.replace(_CODE_PLACEHOLDER, "")))


def _is_own_comment(body: str) -> bool:
    """Whether a comment body looks like one lgtm published on an earlier run.

    Detected here rather than in the git clients so they stay ignorant of how reviews are rendered.
    """
    return _OWN_COMMENT_MARKER in body


def condense_comment_body(body: str, *, max_chars: int = MAX_COMMENT_CHARS) -> str:
    """Reduce a comment body to the part that could actually be repeated."""
    condensed = _HTML_COMMENT.sub("", body)
    condensed = _DETAILS_BLOCK.sub("", condensed)
    condensed = _FENCED_BLOCK.sub(_CODE_PLACEHOLDER, condensed)
    condensed = _BLANK_RUN.sub("\n\n", condensed)
    condensed = condensed.strip()

    if len(condensed) <= max_chars:
        return condensed

    truncated = condensed[:max_chars]
    # Prefer cutting at a line, then a word, so the tail stays readable.
    for separator in ("\n", " "):
        cut = truncated.rfind(separator)
        if cut > max_chars // 2:
            truncated = truncated[:cut]
            break
    return truncated.rstrip() + " […]"


def build_existing_comments_context(
    comments: list[ExistingReviewComment],
    *,
    changed_files: list[str],
    max_comments: int = MAX_COMMENTS,
    max_comment_chars: int = MAX_COMMENT_CHARS,
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> list[ExistingReviewComment]:
    """Select and condense the existing comments worth showing the reviewer agent.

    Comments anchored to a file outside the current diff are dropped: the reviewer cannot comment there, so it
    cannot repeat them. What survives is condensed, then capped by count and by total size.
    """
    if not comments:
        return []

    changed = set(changed_files)
    relevant: list[ExistingReviewComment] = []
    dropped_out_of_diff = 0
    dropped_empty = 0
    for comment in comments:
        if comment.file_path is not None and comment.file_path not in changed:
            dropped_out_of_diff += 1
            continue
        condensed_body = condense_comment_body(comment.body, max_chars=max_comment_chars)
        if not _has_substance(condensed_body):
            dropped_empty += 1
            continue
        relevant.append(
            comment.model_copy(update={"body": condensed_body, "is_own_comment": _is_own_comment(comment.body)})
        )

    # Spend the budget on the comments most likely to be repeated. lgtm's own earlier comments come first — those
    # are the ones it would write again — then human comments, most recent first within each group. The clients
    # return comments in chronological order, so the list index is the recency.
    ranked = sorted(
        enumerate(relevant),
        key=lambda pair: (not pair[1].is_own_comment, -pair[0]),
    )

    selected_indexed: list[tuple[int, ExistingReviewComment]] = []
    total = 0
    dropped_over_count = 0
    dropped_over_budget = 0
    for index, comment in ranked:
        if len(selected_indexed) >= max_comments:
            dropped_over_count += 1
            continue
        size = len(comment.body)
        if total + size > max_total_chars:
            dropped_over_budget += 1
            continue
        total += size
        selected_indexed.append((index, comment))

    # Emit in the order they were posted, so the thread still reads chronologically.
    selected = [comment for _, comment in sorted(selected_indexed, key=lambda pair: pair[0])]

    if dropped_out_of_diff or dropped_empty or dropped_over_count or dropped_over_budget:
        logger.info(
            "Existing comments context: kept %d of %d (dropped %d outside the diff, %d empty after condensing, "
            "%d over the %d-comment limit, %d over the %d-character budget)",
            len(selected),
            len(comments),
            dropped_out_of_diff,
            dropped_empty,
            dropped_over_count,
            max_comments,
            dropped_over_budget,
            max_total_chars,
        )
    else:
        logger.info("Existing comments context: kept all %d comments (%d characters)", len(selected), total)

    return selected
