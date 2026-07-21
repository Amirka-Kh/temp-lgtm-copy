"""Orchestrate the discuss flow: fetch discussion → build prompt → call AI → post reply."""

import logging

from pydantic_ai import Agent

from lgtm_ai.ai.agent import get_ai_model
from lgtm_ai.discuss.prompt import DiscussionThread, build_discuss_prompt
from lgtm_ai.git_client.base import GitClient
from lgtm_ai.git_client.exceptions import PublishGuideError

logger = logging.getLogger("lgtm.discuss")


class DiscussClient:
    """Handles the full discuss command flow."""

    def __init__(self, git_client: GitClient, model_name: str, api_key: str, model_url: str | None = None) -> None:
        self.git_client = git_client
        self.model_name = model_name
        self.api_key = api_key
        self.model_url = model_url

    def discuss(
        self,
        pr_url: object,  # PRUrl instance with source attribute
        discussion_id: str,
        user_message: str,
    ) -> str:
        """Execute the full discuss flow and return the AI response.

        Args:
            pr_url: The parsed PR/MR URL (PRUrl instance)
            discussion_id: The ID of the discussion thread to query
            user_message: The user's question about the review note

        Returns:
            The AI-generated response string
        """
        logger.debug("discuss() got pr_url type=%s repr=%r", type(pr_url), pr_url)
        if isinstance(pr_url, type) or type(pr_url).__name__ == "TypeAliasType":
            raise TypeError(f"Expected a parsed PR URL instance, got the type/alias itself: {pr_url!r}")
        logger.info("Fetching discussion '%s' from %s", discussion_id, pr_url.full_url)
        import traceback as _tb

        try:
            thread = self.git_client.get_discussion(pr_url, discussion_id)
        except Exception:
            logger.error("get_discussion failed:\n%s", _tb.format_exc())
            raise
        logger.info(
            "Discussion fetched: %d notes, file=%s line=%s file_content_len=%s",
            len(thread.notes), thread.file_path, thread.line_number,
            len(thread.file_content) if thread.file_content else None,
        )

        prompt = build_discuss_prompt(thread, user_message)
        logger.info(
            "DEBUG discuss client: prompt length=%d file_content_in_prompt=%s",
            len(prompt), "yes" if thread.file_content else "no",
        )

        logger.info("Calling AI model '%s' for discussion response", self.model_name)
        model = get_ai_model(model_name=self.model_name, api_key=self.api_key, model_url=self.model_url)
        agent = Agent[None, str](deps_type=None, output_type=str)
        raw_res = agent.run_sync(model=model, user_prompt=prompt)
        response = raw_res.output

        logger.info("AI response generated: %d chars", len(response))
        return response

    def reply_to_discussion(
        self,
        pr_url: object,  # PRUrl instance with source attribute
        discussion_id: str,
        response: str,
    ) -> None:
        """Post the AI response as a reply to the discussion thread.

        Args:
            pr_url: The parsed PR/MR URL (PRUrl instance)
            discussion_id: The ID of the discussion thread
            response: The AI-generated response to post
        """
        from lgtm_ai.base.schemas import PRSource

        logger.info("Posting reply to discussion '%s'", discussion_id)

        try:
            if isinstance(pr_url, object):
                source = getattr(pr_url, "source", None)
            else:
                source = None

            if source == PRSource.gitlab:
                self._reply_gitlab(pr_url, discussion_id, response)  # type: ignore[arg-type]
            elif source == PRSource.github:
                self._reply_github(pr_url, discussion_id, response)  # type: ignore[arg-type]
            else:
                logger.warning("Unknown or unsupported source for discussion reply: %s", source)
                raise NotImplementedError(f"Discussion replies not supported for source: {source}")

            logger.info("Reply posted successfully")
        except Exception as err:
            logger.error("Failed to post reply to discussion '%s': %s", discussion_id, err)
            raise PublishGuideError(str(err)) from err

    def _reply_gitlab(self, pr_url: object, discussion_id: str, response: str) -> None:
        """Post a reply inside an existing GitLab MR discussion thread.

        GitLab API: POST /projects/:id/merge_requests/:mr_iid/discussions/:discussion_id/notes
        """
        import gitlab.exceptions

        try:
            gl = self.git_client.client  # type: ignore[attr-defined]
            project_path = getattr(pr_url, "repo_path", None)
            mr_num = getattr(pr_url, "pr_number", None)

            if not project_path or not mr_num:
                raise ValueError("Missing repo_path or pr_number from PRUrl")

            project = gl.projects.get(project_path)
            mr = project.mergerequests.get(mr_num)
            discussion = mr.discussions.get(discussion_id)
            discussion.notes.create({"body": response})
        except gitlab.exceptions.GitlabError as err:
            logger.error("GitLab API error while posting reply: %s", err)
            raise

    def _reply_github(self, pr_url: object, discussion_id: str, response: str) -> None:
        """Post a reply to a GitHub PR (creates a general PR comment since review comments don't support replies)."""
        import github

        from lgtm_ai.git_client.utils import get_git_client as _get_git_client

        try:
            gh = self.git_client.client  # type: ignore[attr-defined]
            repo_path = pr_url.repo_path if hasattr(pr_url, "repo_path") else None
            mr_num = pr_url.pr_number if hasattr(pr_url, "pr_number") else None

            if not repo_path or not mr_num:
                raise ValueError("Missing repo_path or pr_number from PRUrl")

            github_client = github.Github(gh.access_token)  # type: ignore[attr-defined]
            repo = github_client.get_repo(repo_path)
            pr = repo.get_pull(mr_num)

            reply_text = (
                f"**AI Reply to Review Comment #{discussion_id}:**\n\n"
                f"{response}"
            )
            pr.create_issue_comment(reply_text)
        except github.GithubException as err:
            logger.error("GitHub API error while posting reply: %s", err)
            raise
        except Exception as err:
            logger.error("Unexpected error posting GitHub reply: %s", err)
            raise