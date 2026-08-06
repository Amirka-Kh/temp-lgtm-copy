"""Splitting of PR diffs so that they fit in the context window of the AI model.

When a PR is too big to be reviewed in a single call, lgtm splits its diff in as many parts
as needed and reviews them one by one, instead of failing with a "prompt too large" error
and leaving the PR without a review.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Final

from lgtm_ai.ai.context_window import estimate_tokens
from lgtm_ai.git.parser import DiffResult, ModifiedLine

logger = logging.getLogger("lgtm.ai")

TRUNCATED_LINE_SUFFIX: Final = "... [truncated by lgtm]"
"""Marker appended to a changed line that had to be cut short to fit in the context window."""


@dataclass(frozen=True, slots=True)
class DiffChunk:
    """A group of file diffs small enough to be reviewed in a single call to the AI model."""

    diffs: list[DiffResult]
    truncated_files: tuple[str, ...] = field(default=())

    @property
    def file_paths(self) -> tuple[str, ...]:
        return tuple(diff.metadata.new_path for diff in self.diffs)


def serialize_diffs(diffs: list[DiffResult]) -> str:
    """Serialize file diffs to the JSON string representation the AI model receives."""
    return json.dumps([diff.model_dump() for diff in diffs])


def chunk_diffs(diffs: list[DiffResult], *, budget_tokens: int) -> list[DiffChunk]:
    """Group file diffs into chunks that each fit within `budget_tokens`.

    Files are never split across chunks (a reviewer needs to see all the changes of a file
    at once); a single file whose diff does not fit on its own is truncated instead.
    """
    chunks: list[DiffChunk] = []
    current: list[DiffResult] = []
    current_tokens = 0
    current_truncated: list[str] = []

    for diff in diffs:
        fitting_diff, was_truncated = truncate_diff(diff, budget_tokens=budget_tokens)
        diff_tokens = estimate_tokens(serialize_diffs([fitting_diff]))

        if current and current_tokens + diff_tokens > budget_tokens:
            chunks.append(DiffChunk(diffs=current, truncated_files=tuple(current_truncated)))
            current, current_tokens, current_truncated = [], 0, []

        current.append(fitting_diff)
        current_tokens += diff_tokens
        if was_truncated:
            current_truncated.append(fitting_diff.metadata.new_path)

    if current:
        chunks.append(DiffChunk(diffs=current, truncated_files=tuple(current_truncated)))

    return chunks


def truncate_diff(diff: DiffResult, *, budget_tokens: int) -> tuple[DiffResult, bool]:
    """Drop changed lines from the given file diff until it fits within `budget_tokens`.

    Returns the (possibly untouched) diff and whether it had to be truncated.
    """
    if estimate_tokens(serialize_diffs([diff])) <= budget_tokens:
        return diff, False

    overhead = estimate_tokens(serialize_diffs([diff.model_copy(update={"modified_lines": []})]))
    kept: list[ModifiedLine] = []
    used = overhead
    for line in diff.modified_lines:
        line_tokens = estimate_tokens(json.dumps(line.model_dump()))
        if kept and used + line_tokens > budget_tokens:
            break
        kept.append(line)
        used += line_tokens

    if len(kept) == 1 and used > budget_tokens:
        # A single changed line does not fit on its own (minified files, generated code, ...).
        kept = [_truncate_line(kept[0], budget_tokens=max(budget_tokens - overhead, 1))]

    logger.warning(
        "Diff of file %s does not fit in the context window; keeping %d out of %d changed lines",
        diff.metadata.new_path,
        len(kept),
        len(diff.modified_lines),
    )
    return diff.model_copy(update={"modified_lines": kept}), True


def render_diff_payload(chunk: DiffChunk, *, part: int = 1, total_parts: int = 1) -> str:
    """Render the diff of a chunk as it should appear in the prompt, with any caveats."""
    notes: list[str] = []
    if total_parts > 1:
        notes.append(
            f"NOTE: This PR is too large to be reviewed in one go, so its diff has been split in "
            f"{total_parts} parts. This is part {part}, which only contains the changes to the "
            f"following files: {', '.join(chunk.file_paths)}. Review these changes only, and do not "
            f"comment on code you cannot see."
        )
    if chunk.truncated_files:
        notes.append(
            f"NOTE: The diff of the following files did not fit in the context window and was "
            f"truncated, so you are only seeing their first changed lines: "
            f"{', '.join(chunk.truncated_files)}."
        )

    payload = serialize_diffs(chunk.diffs)
    if notes:
        payload = "\n\n".join([payload, *notes])
    return payload


def _truncate_line(line: ModifiedLine, *, budget_tokens: int) -> ModifiedLine:
    """Cut the contents of a single changed line so that it fits in the given budget."""
    max_chars = max(int(budget_tokens * len(line.line) / max(estimate_tokens(json.dumps(line.model_dump())), 1)), 1)
    return line.model_copy(update={"line": line.line[:max_chars] + TRUNCATED_LINE_SUFFIX})
