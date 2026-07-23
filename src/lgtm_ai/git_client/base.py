from typing import TYPE_CHECKING, Protocol

from lgtm_ai.ai.schemas import Review, ReviewGuide
from lgtm_ai.base.schemas import PRUrl
from lgtm_ai.git_client.schemas import ContextBranch, IssueContent, PRDiff, PRMetadata
from pydantic import HttpUrl

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import through the discuss package.
    from lgtm_ai.discuss.prompt import DiscussionThread


class GitClient(Protocol):
    """Interface for any Git service client."""

    def get_diff_from_url(self, pr_url: PRUrl) -> PRDiff:
        """Get the diff from the PR URL. It will be used to generate the review."""

    def publish_review(self, pr_url: PRUrl, review: Review) -> None:
        """Publish a whole review to the PR. It will create several comments/reviews etc. depending on the git service."""

    def get_pr_metadata(self, pr_url: PRUrl) -> PRMetadata:
        """Get metadata for the PR given its URL."""

    def get_issue_content(self, issues_url: HttpUrl, issue_id: str) -> IssueContent | None:
        """Fetch the content of an issue from the base URL of the issues page."""

    def publish_guide(self, pr_url: PRUrl, guide: ReviewGuide) -> None:
        """Publish a review guide to the PR."""

    def get_file_contents(self, pr_url: PRUrl, file_path: str, branch_name: ContextBranch) -> str | None:
        """Get contents of the file from `file_path` from the given branch.

        It should never raise, and instead return None if the file cannot be downloaded.
        """

    def get_discussion(self, pr_url: PRUrl, discussion_id: str) -> "DiscussionThread":
        """Fetch a specific discussion thread (or review comment) with its notes and code context."""
