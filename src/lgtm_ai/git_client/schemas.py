from typing import Literal

from lgtm_ai.git.parser import DiffResult
from pydantic import BaseModel

type ContextBranch = Literal["source", "target"]


class PRDiff(BaseModel):
    id: int
    diff: list[DiffResult]
    changed_files: list[str]
    target_branch: str
    source_branch: str


class PRMetadata(BaseModel):
    title: str
    description: str


class IssueContent(BaseModel):
    title: str
    description: str


class ExistingReviewComment(BaseModel):
    """A comment already present on the PR, used to stop the reviewer repeating what has been said."""

    body: str
    author: str
    is_own_comment: bool = False
    """Whether lgtm itself wrote this comment on an earlier run. Those are the ones it is most likely to repeat."""
    file_path: str | None = None
    """File the comment is anchored to, or None for a PR-level comment."""
    line_number: int | None = None
    is_resolved: bool = False
    """Whether the thread was marked resolved. Resolved points must not be raised again."""
