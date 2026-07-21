import base64
import binascii
import functools
import logging
from typing import Any, cast
from urllib.parse import urlparse

import gitlab
import gitlab.exceptions
import gitlab.v4
import gitlab.v4.objects
import textwrap

from lgtm_ai.ai.schemas import Review, ReviewComment, ReviewGuide
from lgtm_ai.base.schemas import PRUrl
from lgtm_ai.discuss.prompt import DiscussionNote, DiscussionThread
from lgtm_ai.formatters.base import Formatter
from lgtm_ai.git.exceptions import GitDiffParseError
from lgtm_ai.git.parser import DiffFileMetadata, DiffResult, parse_diff_patch
from lgtm_ai.git_client.base import GitClient
from lgtm_ai.git_client.exceptions import (
    InvalidGitAuthError,
    PublishGuideError,
    PublishReviewError,
    PullRequestDiffError,
    PullRequestDiffNotFoundError,
    PullRequestMetadataError,
)
from lgtm_ai.git_client.schemas import ContextBranch, IssueContent, PRDiff, PRMetadata
from pydantic import HttpUrl

logger = logging.getLogger("lgtm.git")


class GitlabClient(GitClient):
    def __init__(self, client: gitlab.Gitlab, formatter: Formatter[str]) -> None:
        self.client = client
        self.formatter = formatter
        self._pr: gitlab.v4.objects.ProjectMergeRequest | None = None

    def get_diff_from_url(self, pr_url: PRUrl) -> PRDiff:
        """Return a PRDiff object containing an identifier to the diff and a stringified representation of the diff from latest version of the given pull request URL."""
        try:
            self.client.auth()
            logger.info("Authenticated with GitLab")
        except gitlab.exceptions.GitlabAuthenticationError as err:
            logger.error("Invalid GitLab authentication token")
            raise InvalidGitAuthError from err

        logger.info("Fetching diff from GitLab")
        try:
            pr = _get_pr_from_url(self.client, pr_url)
            diff = self._get_diff_from_pr(pr)
        except gitlab.exceptions.GitlabError as err:
            logger.error("Failed to retrieve the diff of the pull request")
            raise PullRequestDiffError from err

        return PRDiff(
            id=diff.id,
            diff=self._parse_gitlab_git_diff(diff.diffs),
            changed_files=[change["new_path"] for change in diff.diffs],
            target_branch=pr.target_branch,
            source_branch=pr.source_branch,
        )

    def get_pr_metadata(self, pr_url: PRUrl) -> PRMetadata:
        try:
            pr = _get_pr_from_url(self.client, pr_url)
        except gitlab.exceptions.GitlabError as err:
            logger.error("Failed to retrieve the metadata of the pull request")
            raise PullRequestMetadataError from err
        return PRMetadata(
            title=pr.title or "",
            description=pr.description or "",
        )

    def get_issue_content(self, issues_url: HttpUrl, issue_id: str) -> IssueContent | None:
        try:
            project = _get_project_from_issues_url(self.client, issues_url)
            issue = project.issues.get(issue_id)
        except (gitlab.exceptions.GitlabError, ValueError):
            logger.warning("Failed to retrieve the issue content from GitLab for issue %s", issue_id)
            return None

        return IssueContent(
            title=issue.title or "",
            description=issue.description or "",
        )

    def publish_review(self, pr_url: PRUrl, review: Review) -> None:
        logger.info("Publishing review to GitLab")
        try:
            pr = _get_pr_from_url(self.client, pr_url)
            self._post_review_summary(pr, review)
            failed_comments = self._post_review_comments(pr, review)
            logger.warning(
                "Some comments could not be posted to GitLab; total: %d, failed: %d",
                len(review.review_response.comments),
                len(failed_comments),
            )
        except gitlab.exceptions.GitlabError as err:
            raise PublishReviewError from err

    def publish_guide(self, pr_url: PRUrl, guide: ReviewGuide) -> None:
        try:
            pr = _get_pr_from_url(self.client, pr_url)
            pr.notes.create({"body": self.formatter.format_guide(guide)})
        except gitlab.exceptions.GitlabError as err:
            raise PublishGuideError from err

    def get_file_contents(self, pr_url: PRUrl, file_path: str, branch_name: ContextBranch) -> str | None:
        project = _get_project_from_url(self.client, pr_url.repo_path)
        pr = _get_pr_from_url(self.client, pr_url)
        ref = pr.sha if branch_name == "source" else pr.target_branch
        logger.info(
            "DEBUG get_file_contents: fetching file — path=%s ref_type=%s ref=%s (pr.sha=%s pr.target_branch=%s)",
            file_path, str(branch_name), ref, pr.sha, pr.target_branch,
        )
        try:
            file = project.files.get(
                file_path=file_path,
                ref=ref,
            )
            logger.info("DEBUG get_file_contents: GitLab API returned file object (content len=%d)", len(file.content or ""))
        except gitlab.exceptions.GitlabError as err:
            logger.warning(
                "DEBUG get_file_contents: GitLab API failed for path=%s ref=%s — error=%s",
                file_path, ref, err,
            )
            return None

        try:
            content = base64.b64decode(file.content).decode()
            logger.info("DEBUG get_file_contents: decoded file %s successfully (%d chars)", file_path, len(content))
        except (binascii.Error, UnicodeDecodeError) as err:
            logger.warning(
                "DEBUG get_file_contents: decode failed for %s — error=%s",
                file_path, err,
            )
            return None
        return content

    def get_discussion(self, pr_url: PRUrl, discussion_id: str) -> DiscussionThread:
        """Fetch a specific discussion thread from the MR.

        Returns a DiscussionThread with all notes and code context.
        """
        try:
            project = _get_project_from_url(self.client, pr_url.repo_path)
            pr = project.mergerequests.get(pr_url.pr_number)
            discussion = pr.discussions.get(discussion_id)
        except gitlab.exceptions.GitlabError as err:
            logger.error("Failed to retrieve discussion %s from MR %s", discussion_id, pr_url.full_url)
            raise

        notes: list[DiscussionNote] = []
        file_path: str | None = None
        line_number: int | None = None

        # Extract notes from the discussion thread.
        # python-gitlab's MR discussion note manager is inconsistent across versions:
        #   - .list() may not exist at all (some versions)
        #   - notes may be embedded in the response data directly
        # We try multiple access patterns to be robust.
        raw_notes = self._fetch_discussion_notes(discussion, project, discussion_id)

        for note_data in raw_notes:
            if note_data.get("system"):
                continue  # Skip system notes (e.g., "merged", "changed target branch")
            author = note_data["author"]["name"] if note_data.get("author") else "Unknown"
            created_at = str(note_data.get("created_at", "unknown"))
            body = note_data.get("body") or ""
            is_position = bool(note_data.get("position"))

            # Extract code position from the first position note
            if is_position and not file_path:
                position = note_data["position"]
                file_path = position.get("new_path") or position.get("old_path")
                line_number = position.get("new_line") or position.get("old_line")

            notes.append(DiscussionNote(body, author, created_at, is_position))

        # Get full file content for complete context (cross-references, calls, etc.)
        file_content: str | None = None
        if file_path:
            logger.info(
                "DEBUG get_discussion: fetching file content — path=%s line=%d branch=source",
                file_path, line_number,
            )
            import traceback as _tb

            try:
                logger.debug("DEBUG get_discussion: calling get_file_contents — pr_url type=%r pr_url.source=%r", type(pr_url), getattr(pr_url, "source", "<NO SOURCE ATTR>"))
                file_content = self.get_file_contents(pr_url, file_path, "source")
            except Exception:
                logger.error("get_file_contents failed for %s:\n%s", file_path, _tb.format_exc())
                raise
            logger.info(
                "DEBUG get_discussion: file content fetch result — path=%s file_content_len=%s",
                file_path, len(file_content) if file_content else None,
            )
        else:
            logger.warning("DEBUG get_discussion: no file_path extracted from notes — file_content will be None")

        return DiscussionThread(notes, file_path, line_number, file_content)

    def _fetch_discussion_notes(
        self, discussion: Any, project: Any, discussion_id: str
    ) -> list[dict[str, Any]]:
        """Return the raw note dicts embedded in a fetched MR discussion."""
        notes = discussion.attributes.get("notes")
        if notes:
            logger.debug("Discussion %s: %d embedded notes", discussion_id, len(notes))
            return notes

        logger.warning("No notes found in discussion %s", discussion_id)
        return []
    
    def _parse_gitlab_git_diff(self, diffs: list[dict[str, object]]) -> list[DiffResult]:
        parsed_diffs: list[DiffResult] = []
        for diff in diffs:
            try:
                diff_text = diff.get("diff")
                if diff_text is None:
                    logger.warning("Diff text is empty, skipping..., diff: %s", diff)
                    continue
                parsed = parse_diff_patch(
                    metadata=DiffFileMetadata.model_validate(diff),
                    diff_text=cast(str, diff_text),
                )
            except GitDiffParseError:
                logger.exception(
                    "Failed to parse diff patch for file %s, will skip it", diff.get("new_path", "unknown")
                )
                continue
            parsed_diffs.append(parsed)

        return parsed_diffs

    def _post_review_summary(self, pr: gitlab.v4.objects.ProjectMergeRequest, review: Review) -> None:
        pr.notes.create({"body": self.formatter.format_review_summary_section(review)})

    def _post_review_comments(self, pr: gitlab.v4.objects.ProjectMergeRequest, review: Review) -> list[ReviewComment]:
        """Post comments on the file & filenumber they refer to.

        The AI currently makes mistakes which make gitlab fail to accurately post a comment.
        For example with the line number a comment refers to (whether it's a line on the 'old' file vs the 'new file).
        To avoid blocking the review, we try once with `new_line`, retry with `old_line`, then try to post the comment on the file level and finally return the comments to be posted with the main summary.

        Returns:
            list[ReviewComment]: list of comments that could not be created, and therefore should be appended to the review summary
        """
        logger.info("Posting comments to GitLab")
        failed_comments: list[ReviewComment] = []

        diff = pr.diffs.get(review.pr_diff.id)
        for review_comment in review.review_response.comments:
            position = {
                "base_sha": diff.base_commit_sha,
                "head_sha": diff.head_commit_sha,
                "start_sha": diff.start_commit_sha,
                "new_path": review_comment.new_path,
                "old_path": review_comment.old_path,
                "position_type": "text",
            }
            if review_comment.is_comment_on_new_path:
                position["new_line"] = review_comment.line_number
            else:
                position["old_line"] = review_comment.line_number

            gitlab_comment = {
                "body": self.formatter.format_review_comment(review_comment),
                "position": position,
            }

            comment_create_success = self._attempt_comment_at_positions(pr, gitlab_comment)

            if not comment_create_success:
                # Add it to the list of failed comments to be published in the summary comment
                failed_comments.append(review_comment)

        if failed_comments:
            logger.warning(
                "Some comments could not be posted to GitLab; total: %d, failed: %d",
                len(review.review_response.comments),
                len(failed_comments),
            )
        return failed_comments

    def _attempt_comment_at_positions(
        self, pr: gitlab.v4.objects.ProjectMergeRequest, gitlab_comment: dict[str, Any]
    ) -> bool:
        """Try to post comments at decreasingly specific positions.

        By default we want to just try original target, then swap lines, then post to file, then give up.

        Returns whether any of the attempts were successful.
        """
        comment_create_success: bool = True
        try:
            pr.discussions.create(gitlab_comment)
        except gitlab.exceptions.GitlabError:
            comment_create_success = False

        position = gitlab_comment["position"]
        if not comment_create_success:
            # Switch new_line <-> old_line in case the AI made a mistake with `is_comment_on_new_path`
            logger.debug("Failed to post comment, retrying with new_line <-> old_line")
            if "old_line" in position:
                position["new_line"] = position.pop("old_line")
            else:
                position["old_line"] = position.pop("new_line")

            comment_create_success = True
            try:
                pr.discussions.create(gitlab_comment)
            except gitlab.exceptions.GitlabError:
                comment_create_success = False

        if not comment_create_success:
            # Failed to attach to a line, so let's try at file level
            logger.debug("Failed to post for neither line, retrying with a file-level comment")
            _ = position.pop("new_line", None)
            _ = position.pop("old_line", None)
            position["position_type"] = "file"

            comment_create_success = True
            try:
                pr.discussions.create(gitlab_comment)
            except gitlab.exceptions.GitlabError:
                comment_create_success = False
                logger.debug(
                    "Failed to post the comment anywhere specific, it will go to general description (hopefully)"
                )

        return comment_create_success

    def _get_diff_from_pr(self, pr: gitlab.v4.objects.ProjectMergeRequest) -> gitlab.v4.objects.ProjectMergeRequestDiff:
        """Gitlab returns multiple "diff" objects for a single MR, which correspond to each pushed "version" of the MR.

        We only need to review the latest one, which is the first in the list.
        """
        try:
            latest_diff = next(iter(pr.diffs.list()))
        except StopIteration as err:
            raise PullRequestDiffNotFoundError from err

        return pr.diffs.get(latest_diff.id)


@functools.lru_cache(maxsize=32)
def _get_pr_from_url(client: gitlab.Gitlab, pr_url: PRUrl) -> gitlab.v4.objects.ProjectMergeRequest:
    logger.debug("Fetching mr from GitLab (cache miss)")
    project = _get_project_from_url(client, pr_url.repo_path)
    return project.mergerequests.get(pr_url.pr_number)


@functools.lru_cache(maxsize=32)
def _get_project_from_url(client: gitlab.Gitlab, repo_path: str) -> gitlab.v4.objects.Project:
    """Get the project from the GitLab client using the project path from the PR URL."""
    logger.debug("Fetching project from GitLab (cache miss)")
    return client.projects.get(repo_path)


def _get_project_from_issues_url(client: gitlab.Gitlab, issues_url: HttpUrl) -> gitlab.v4.objects.Project:
    """Get the project from the GitLab client using the project path from the issues URL."""
    parsed = urlparse(str(issues_url))
    project_path, _ = parsed.path.split("/-/issues")
    return _get_project_from_url(client, project_path.strip("/"))
