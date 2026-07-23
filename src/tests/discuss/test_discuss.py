from unittest import mock

import pytest
from lgtm_ai.base.schemas import PRSource, PRUrl
from lgtm_ai.discuss.client import DiscussClient
from lgtm_ai.discuss.prompt import DiscussionNote, DiscussionThread, build_discuss_prompt
from lgtm_ai.git_client.exceptions import PublishGuideError
from pydantic_ai.models.test import TestModel
from tests.review.utils import MockGitClient

PR_URL = PRUrl(
    full_url="https://gitlab.com/foo/bar/-/merge_requests/1",
    base_url="https://gitlab.com",
    repo_path="foo/bar",
    pr_number=1,
    source=PRSource.gitlab,
)


def _thread(**kwargs: object) -> DiscussionThread:
    defaults: dict[str, object] = {
        "notes": [
            DiscussionNote("original review note", "lgtm", "2025-01-01", is_position_note=True),
            DiscussionNote("a reply", "author", "2025-01-02"),
        ],
        "file_path": "file.py",
        "line_number": 2,
        "file_content": "line one\nline two\nline three",
    }
    defaults.update(kwargs)
    return DiscussionThread(**defaults)  # type: ignore[arg-type]


class TestBuildDiscussPrompt:
    def test_full_thread_includes_all_sections(self) -> None:
        prompt = build_discuss_prompt(_thread(), "why is this wrong?")

        assert "original review note" in prompt
        assert "```file.py" in prompt
        assert "The review comment targets line 2." in prompt
        assert ">>> line two" in prompt
        assert "[author] (2025-01-02):" in prompt
        assert "a reply" in prompt
        assert "why is this wrong?" in prompt

    def test_thread_without_file_content(self) -> None:
        prompt = build_discuss_prompt(_thread(file_content=None), "question")

        assert "full source file" not in prompt
        assert "question" in prompt

    def test_thread_with_line_number_out_of_range(self) -> None:
        prompt = build_discuss_prompt(_thread(line_number=99), "question")

        assert "The review comment targets line 99." in prompt
        assert ">>>" not in prompt

    def test_empty_thread_uses_fallback_prompt(self) -> None:
        prompt = build_discuss_prompt(_thread(notes=[]), "question")

        assert "empty or has no retrievable notes" in prompt
        assert "```file.py" in prompt
        assert "question" in prompt

    def test_empty_thread_without_file_content(self) -> None:
        prompt = build_discuss_prompt(_thread(notes=[], file_content=None), "question")

        assert "empty or has no retrievable notes" in prompt
        assert "```" in prompt


class TestDiscussClient:
    def test_discuss_returns_ai_response(self) -> None:
        client = DiscussClient(git_client=MockGitClient(), model_name="some-model", api_key="key")

        with mock.patch("lgtm_ai.discuss.client.get_ai_model", return_value=TestModel()):
            response = client.discuss(pr_url=PR_URL, discussion_id="1", user_message="what does this mean?")

        assert response

    def test_discuss_propagates_git_client_errors(self) -> None:
        git_client = MockGitClient()
        git_client.get_discussion = mock.Mock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        client = DiscussClient(git_client=git_client, model_name="some-model", api_key="key")

        with pytest.raises(RuntimeError, match="boom"):
            client.discuss(pr_url=PR_URL, discussion_id="1", user_message="question")

    def test_reply_to_discussion_unsupported_source(self) -> None:
        client = DiscussClient(git_client=MockGitClient(), model_name="some-model", api_key="key")
        local_url = PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.local)

        with pytest.raises(PublishGuideError, match="not supported"):
            client.reply_to_discussion(pr_url=local_url, discussion_id="1", response="reply")

    def test_reply_to_discussion_wraps_errors(self) -> None:
        client = DiscussClient(git_client=MockGitClient(), model_name="some-model", api_key="key")

        with (
            mock.patch.object(client, "_reply_gitlab", side_effect=RuntimeError("boom")),
            pytest.raises(PublishGuideError, match="boom"),
        ):
            client.reply_to_discussion(pr_url=PR_URL, discussion_id="1", response="reply")

    def test_reply_to_discussion_dispatches_to_gitlab(self) -> None:
        client = DiscussClient(git_client=MockGitClient(), model_name="some-model", api_key="key")

        with mock.patch.object(client, "_reply_gitlab") as reply_gitlab:
            client.reply_to_discussion(pr_url=PR_URL, discussion_id="1", response="reply")

        reply_gitlab.assert_called_once_with(PR_URL, "1", "reply")
