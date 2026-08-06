import json
import textwrap
from typing import Literal
from unittest import mock

import pytest
from lgtm_ai.ai.agent import get_reviewer_agent_with_settings, get_summarizing_agent_with_settings
from lgtm_ai.ai.schemas import AdditionalContext, PublishMetadata, Review, ReviewComment, ReviewResponse
from lgtm_ai.base.exceptions import NothingToReviewError
from lgtm_ai.base.schemas import PRSource, PRUrl
from lgtm_ai.config.constants import DEFAULT_AI_MODEL
from lgtm_ai.config.handler import ResolvedConfig
from lgtm_ai.git.parser import DiffFileMetadata, DiffResult, ModifiedLine
from lgtm_ai.git_client.schemas import PRDiff
from lgtm_ai.review import CodeReviewer
from lgtm_ai.review.context import ContextRetriever
from lgtm_ai.review.reviewer import _merge_review_responses
from lgtm_ai.review.exceptions import (
    ClientUsageLimitsExceededError,
    InvalidAIResponseError,
    ServerError,
    ServerUsageLimitsExceededError,
    UnknownAIError,
)
from pydantic import ValidationError
from pydantic_ai import (
    AgentRunError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    capture_run_messages,
    models,
)
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from tests.review.utils import MOCK_DIFF, MockGitClient

# This is a safety measure to make sure we don't accidentally make real requests to the LLM while testing,
# see ALLOW_MODEL_REQUESTS for more details.
models.ALLOW_MODEL_REQUESTS = False


def _get_ai_validation_error(*, is_validation_error: bool) -> UnexpectedModelBehavior:
    exc = UnexpectedModelBehavior("Error")
    exc_lvl2 = Exception("Error level 2")
    exc_lvl3 = ValidationError("Error level 3", []) if is_validation_error else Exception("Error level 3")
    exc_lvl2.__context__ = exc_lvl3
    exc.__context__ = exc_lvl2

    return exc


@pytest.fixture
def context_retriever() -> ContextRetriever:
    return ContextRetriever(
        git_client=MockGitClient(),
        issues_client=MockGitClient(),
        httpx_client=mock.Mock(),
    )


def test_get_review_from_url_valid(context_retriever: ContextRetriever) -> None:
    test_agent = get_reviewer_agent_with_settings()
    test_summary_agent = get_summarizing_agent_with_settings()
    config = ResolvedConfig(
        ai_api_key="",
        git_api_key="",
        additional_context=(
            AdditionalContext(
                file_url=None,
                prompt="These are the development guidelines for the project. Please follow them.",
                context="contents-of-dev-guidelines",
            ),
            AdditionalContext(
                file_url=None,
                prompt="Yet another prompt",
                context="yet-another-context",
            ),
        ),
    )
    with (
        test_agent.override(
            model=TestModel(),
        ),
        test_summary_agent.override(
            model=TestModel(),
        ),
        capture_run_messages() as messages,
    ):
        code_reviewer = CodeReviewer(
            reviewer_agent=test_agent,
            summarizing_agent=test_summary_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=MockGitClient(),
            context_retriever=context_retriever,
            config=ResolvedConfig(
                ai_api_key="",
                git_api_key="",
                additional_context=(
                    AdditionalContext(
                        file_url=None,
                        prompt="These are the development guidelines for the project. Please follow them.",
                        context="contents-of-dev-guidelines",
                    ),
                    AdditionalContext(
                        file_url=None,
                        prompt="Yet another prompt",
                        context="yet-another-context",
                    ),
                ),
            ),
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

    # We get an actual review object
    assert review == Review(
        pr_diff=PRDiff(
            id=1,
            diff=MOCK_DIFF,
            changed_files=["file-1.txt", "file-2.txt"],
            target_branch="main",
            source_branch="feature",
        ),
        review_response=ReviewResponse(summary="a", raw_score=1),
        metadata=PublishMetadata(
            model_name=DEFAULT_AI_MODEL,
            usage=review.metadata.usage,
            config=config.model_dump(),
        ),
    )

    # There are messages with the correct prompts to the AI agent
    expected_message = (
        textwrap.dedent(
            f"""
        PR METADATA:
        - Title: feat(#288): a title
        - Description: bar



        PR DIFF:
        ```
        {json.dumps([diff.model_dump() for diff in MOCK_DIFF])}
        ```


        CONTEXT:

        ```file=file-1.txt, branch=source
        contents-of-file-1.txt-context
        ```

        ```file=file-2.txt, branch=source
        contents-of-file-2.txt-context
        ```

        ADDITIONAL CONTEXT:

        ```file=None, prompt=These are the development guidelines for the project. Please follow them.
        contents-of-dev-guidelines
        ```

        ```file=None, prompt=Yet another prompt
        yet-another-context
        ```
        """
        ).strip()
        + "\n"
    )
    _assert_agent_message(
        messages,
        expected_message,
        expected_messages=2,
        expected_prompts=["system-prompt", "user-prompt"],
    )


def test_get_review_with_issue(context_retriever: ContextRetriever) -> None:
    test_agent = get_reviewer_agent_with_settings()
    test_summary_agent = get_summarizing_agent_with_settings()
    with (
        test_agent.override(
            model=TestModel(),
        ),
        test_summary_agent.override(
            model=TestModel(),
        ),
        capture_run_messages() as messages,
    ):
        code_reviewer = CodeReviewer(
            reviewer_agent=test_agent,
            summarizing_agent=test_summary_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=MockGitClient(),
            context_retriever=context_retriever,
            config=ResolvedConfig(
                ai_api_key="",
                git_api_key="",
                additional_context=(
                    AdditionalContext(
                        file_url=None,
                        prompt="These are the development guidelines for the project. Please follow them.",
                        context="contents-of-dev-guidelines",
                    ),
                    AdditionalContext(
                        file_url=None,
                        prompt="Yet another prompt",
                        context="yet-another-context",
                    ),
                ),
                issues_platform="gitlab",
                issues_url="https://gitlab.com/example/repo/-/issues/",
            ),
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

    assert isinstance(review, Review)

    # There are messages with the correct prompts to the AI agent
    expected_message = (
        textwrap.dedent(
            f"""
        PR METADATA:
        - Title: feat(#288): a title
        - Description: bar


        USER STORY:
        - Title: Issue title
        - Description: Issue description


        PR DIFF:
        ```
        {json.dumps([diff.model_dump() for diff in MOCK_DIFF])}
        ```


        CONTEXT:

        ```file=file-1.txt, branch=source
        contents-of-file-1.txt-context
        ```

        ```file=file-2.txt, branch=source
        contents-of-file-2.txt-context
        ```

        ADDITIONAL CONTEXT:

        ```file=None, prompt=These are the development guidelines for the project. Please follow them.
        contents-of-dev-guidelines
        ```

        ```file=None, prompt=Yet another prompt
        yet-another-context
        ```
        """
        ).strip()
        + "\n"
    )
    _assert_agent_message(
        messages,
        expected_message,
        expected_messages=2,
        expected_prompts=["system-prompt", "user-prompt"],
    )


def test_summarizing_message_in_review(context_retriever: ContextRetriever) -> None:
    test_agent = mock.Mock()
    test_summarizing_agent = get_summarizing_agent_with_settings()
    test_agent.run_sync.return_value = mock.Mock(
        output=ReviewResponse(summary="a", raw_score=1),
        usage=RunUsage(requests=1, input_tokens=1041, output_tokens=6),
    )

    with (
        test_summarizing_agent.override(
            model=TestModel(),
        ),
        capture_run_messages() as messages,
    ):
        code_reviewer = CodeReviewer(
            reviewer_agent=test_agent,
            summarizing_agent=test_summarizing_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=MockGitClient(),
            context_retriever=context_retriever,
            config=ResolvedConfig(ai_api_key="", git_api_key=""),
        )
        code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

    expected_review_json = ReviewResponse(summary="a", raw_score=1).model_dump_json(indent=2)
    expected_message = "\n".join(
        (
            "PR METADATA:",
            "- Title: feat(#288): a title",
            "- Description: bar",
            "",
            "PR DIFF:",
            "```",
            json.dumps([diff.model_dump() for diff in MOCK_DIFF]),
            "```",
            "",
            "REVIEW:",
            "```",
            expected_review_json,
            "```",
        )
    )
    _assert_agent_message(
        messages,
        expected_message,
        expected_messages=2,
        expected_prompts=["system-prompt", "user-prompt"],
    )


def test_get_review_adds_technologies_to_prompt(context_retriever: ContextRetriever) -> None:
    test_agent = get_reviewer_agent_with_settings()
    test_summary_agent = get_summarizing_agent_with_settings()
    with (
        test_agent.override(
            model=TestModel(),
        ),
        test_summary_agent.override(
            model=TestModel(),
        ),
        capture_run_messages() as messages,
    ):
        code_reviewer = CodeReviewer(
            reviewer_agent=test_agent,
            summarizing_agent=test_summary_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=MockGitClient(),
            context_retriever=context_retriever,
            config=ResolvedConfig(ai_api_key="", git_api_key="", technologies=("COBOL", "FORTRAN", "ODIN")),
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

    assert review
    assert messages

    requests = _get_requests_from_messages(messages)
    assert requests
    first_request = requests[0]
    assert len(first_request.parts) == 2

    content = first_request.parts[0].content
    assert isinstance(content, str)
    assert 'You are an expert in "COBOL", "FORTRAN", "ODIN".' in content


def test_get_review_adds_categories_to_prompt(context_retriever: ContextRetriever) -> None:
    test_agent = get_reviewer_agent_with_settings()
    test_summary_agent = get_summarizing_agent_with_settings()
    with (
        test_agent.override(
            model=TestModel(),
        ),
        test_summary_agent.override(
            model=TestModel(),
        ),
        capture_run_messages() as messages,
    ):
        code_reviewer = CodeReviewer(
            reviewer_agent=test_agent,
            summarizing_agent=test_summary_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=MockGitClient(),
            context_retriever=context_retriever,
            config=ResolvedConfig(ai_api_key="", git_api_key="", categories=("Correctness", "Quality")),
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

    assert review
    assert messages

    requests = _get_requests_from_messages(messages)
    assert requests
    first_request = requests[0]
    assert len(first_request.parts) == 2

    content = first_request.parts[0].content
    assert isinstance(content, str)

    # These two categories are in the prompt
    assert "Correctness" in content
    assert "Quality" in content
    assert (
        "The categories you should exclusively focus on for your review comments are: Correctness, Quality" in content
    )


def test_review_fails_if_all_files_are_excluded() -> None:
    code_reviewer = CodeReviewer(
        reviewer_agent=mock.Mock(),
        summarizing_agent=mock.Mock(),
        model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
        git_client=MockGitClient(),
        context_retriever=ContextRetriever(
            git_client=MockGitClient(),
            issues_client=MockGitClient(),
            httpx_client=mock.Mock(),
        ),
        config=ResolvedConfig(ai_api_key="", git_api_key="", exclude=("*.txt",)),  # we exclude all txt files
    )
    with pytest.raises(NothingToReviewError):
        code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )


def test_file_is_excluded_from_prompt(context_retriever: ContextRetriever) -> None:
    test_agent = get_reviewer_agent_with_settings()
    test_summary_agent = get_summarizing_agent_with_settings()
    with (
        test_agent.override(
            model=TestModel(),
        ),
        test_summary_agent.override(
            model=TestModel(),
        ),
        capture_run_messages() as messages,
    ):
        code_reviewer = CodeReviewer(
            reviewer_agent=test_agent,
            summarizing_agent=test_summary_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=MockGitClient(),
            context_retriever=context_retriever,
            config=ResolvedConfig(ai_api_key="", git_api_key="", exclude=("file2.txt",)),
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

    assert review

    assert any("contents-of-file1" in str(message) for message in messages)
    assert not any("contents-of-file2" in str(message) for message in messages)


@pytest.mark.parametrize(
    ("raised_error", "expected_error"),
    [
        (AgentRunError("Error"), UnknownAIError),
        (ModelHTTPError(500, "Error"), ServerError),
        (ModelHTTPError(429, "Error"), ServerUsageLimitsExceededError),
        (ModelHTTPError(312, "Error"), UnknownAIError),
        (_get_ai_validation_error(is_validation_error=True), InvalidAIResponseError),
        (_get_ai_validation_error(is_validation_error=False), UnknownAIError),
        (UsageLimitExceeded("Error"), ClientUsageLimitsExceededError),
    ],
)
def test_errors_are_handled_on_reviewer_agent(raised_error: Exception, expected_error: type[Exception]) -> None:
    error_agent = mock.Mock()
    error_agent.run_sync.side_effect = raised_error

    code_reviewer = CodeReviewer(
        reviewer_agent=error_agent,
        summarizing_agent=mock.Mock(),
        model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
        git_client=MockGitClient(),
        context_retriever=ContextRetriever(
            git_client=MockGitClient(),
            issues_client=MockGitClient(),
            httpx_client=mock.Mock(),
        ),
        config=ResolvedConfig(ai_api_key="", git_api_key=""),
    )

    with pytest.raises(expected_error):
        code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )


class TestLargePullRequests:
    """Reviews of PRs that do not fit in the context window of the model."""

    def _get_reviewer(
        self,
        *,
        reviewer_agent: mock.Mock,
        summarizing_agent: mock.Mock,
        git_client: MockGitClient,
        ai_context_window: int,
    ) -> CodeReviewer:
        return CodeReviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=summarizing_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=git_client,
            context_retriever=ContextRetriever(
                git_client=git_client,
                issues_client=git_client,
                httpx_client=mock.Mock(),
            ),
            config=ResolvedConfig(ai_api_key="", git_api_key="", ai_context_window=ai_context_window),
        )

    def _get_agent_mock(self, *responses: ReviewResponse) -> mock.Mock:
        agent = mock.Mock()
        agent.run_sync.side_effect = [
            mock.Mock(output=response, usage=RunUsage(requests=1, input_tokens=10, output_tokens=1))
            for response in responses
        ]
        return agent

    def _get_prompts(self, agent: mock.Mock) -> list[str]:
        return [call.kwargs["user_prompt"] for call in agent.run_sync.call_args_list]

    def test_code_context_is_dropped_when_the_prompt_does_not_fit(self) -> None:
        reviewer_agent = self._get_agent_mock(ReviewResponse(summary="a", raw_score=3))
        summarizing_agent = self._get_agent_mock(ReviewResponse(summary="final", raw_score=3))

        code_reviewer = self._get_reviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=summarizing_agent,
            git_client=MockGitClient(file_contents_size=50_000),
            ai_context_window=4_000,
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

        assert review.review_response.summary == "final"
        # The PR is still reviewed in one go, but without the contents of the changed files
        assert reviewer_agent.run_sync.call_count == 1
        prompt = self._get_prompts(reviewer_agent)[0]
        assert "CONTEXT:" not in prompt
        assert "contents-of-file-1.txt-context" not in prompt

    def test_large_prs_are_reviewed_in_several_parts(self) -> None:
        reviewer_agent = self._get_agent_mock(
            ReviewResponse(summary="first part", raw_score=4),
            ReviewResponse(summary="second part", raw_score=2),
        )
        summarizing_agent = self._get_agent_mock(ReviewResponse(summary="final", raw_score=2))

        code_reviewer = self._get_reviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=summarizing_agent,
            git_client=MockGitClient(diff=_get_large_diff(files=4, lines=20, line_length=200)),
            ai_context_window=8_000,
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

        # The review is not lost: the PR is split in parts, each one is reviewed, and they are merged
        assert reviewer_agent.run_sync.call_count == 2
        prompts = self._get_prompts(reviewer_agent)
        assert "part 1" in prompts[0]
        assert "part 2" in prompts[1]
        assert review.review_response.summary == "final"

        # The summarizing agent gets the merged review of all the parts
        summarizing_prompt = self._get_prompts(summarizing_agent)[0]
        assert "first part" in summarizing_prompt
        assert "second part" in summarizing_prompt

    def test_partial_review_is_kept_if_usage_limits_are_hit_while_reviewing_parts(self) -> None:
        reviewer_agent = mock.Mock()
        reviewer_agent.run_sync.side_effect = [
            mock.Mock(
                output=ReviewResponse(summary="first part", raw_score=4),
                usage=RunUsage(requests=1, input_tokens=10, output_tokens=1),
            ),
            UsageLimitExceeded("Limit reached"),
        ]
        summarizing_agent = self._get_agent_mock(ReviewResponse(summary="final", raw_score=4))

        code_reviewer = self._get_reviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=summarizing_agent,
            git_client=MockGitClient(diff=_get_large_diff(files=4, lines=20, line_length=200)),
            ai_context_window=8_000,
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

        assert review.review_response.summary == "final"
        assert "first part" in self._get_prompts(summarizing_agent)[0]

    def test_review_fails_if_no_part_of_the_pr_can_be_reviewed(self) -> None:
        reviewer_agent = mock.Mock()
        reviewer_agent.run_sync.side_effect = UsageLimitExceeded("Limit reached")

        code_reviewer = self._get_reviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=mock.Mock(),
            git_client=MockGitClient(diff=_get_large_diff(files=4, lines=20, line_length=200)),
            ai_context_window=8_000,
        )

        with pytest.raises(ClientUsageLimitsExceededError):
            code_reviewer.review(
                target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
            )

    def test_additional_context_is_dropped_when_it_takes_up_the_whole_context_window(self) -> None:
        reviewer_agent = self._get_agent_mock(*[ReviewResponse(summary="a", raw_score=3)] * 4)
        summarizing_agent = self._get_agent_mock(ReviewResponse(summary="final", raw_score=3))

        code_reviewer = CodeReviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=summarizing_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=MockGitClient(diff=_get_large_diff(files=4, lines=20, line_length=200)),
            context_retriever=ContextRetriever(
                git_client=MockGitClient(), issues_client=MockGitClient(), httpx_client=mock.Mock()
            ),
            config=ResolvedConfig(
                ai_api_key="",
                git_api_key="",
                ai_context_window=8_000,
                additional_context=(AdditionalContext(file_url=None, prompt="guidelines", context="x" * 30_000),),
            ),
        )
        code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

        assert reviewer_agent.run_sync.call_count > 1
        assert all("ADDITIONAL CONTEXT:" not in prompt for prompt in self._get_prompts(reviewer_agent))

    def test_summarizing_step_is_skipped_when_the_review_does_not_fit(self) -> None:
        reviewer_agent = self._get_agent_mock(
            ReviewResponse(summary="a" * 40_000, raw_score=3), ReviewResponse(summary="b" * 40_000, raw_score=2)
        )
        summarizing_agent = mock.Mock()

        code_reviewer = self._get_reviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=summarizing_agent,
            git_client=MockGitClient(diff=_get_large_diff(files=4, lines=20, line_length=200)),
            ai_context_window=8_000,
        )
        review = code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

        # The merged review of all the parts is published as is instead of failing
        summarizing_agent.run_sync.assert_not_called()
        assert review.review_response.summary.startswith("a" * 100)
        assert review.review_response.raw_score == 2

    def test_diff_is_omitted_from_the_summarizing_prompt_when_it_does_not_fit(self) -> None:
        reviewer_agent = self._get_agent_mock(
            ReviewResponse(summary="a", raw_score=3), ReviewResponse(summary="b", raw_score=3)
        )
        summarizing_agent = self._get_agent_mock(ReviewResponse(summary="final", raw_score=3))

        code_reviewer = self._get_reviewer(
            reviewer_agent=reviewer_agent,
            summarizing_agent=summarizing_agent,
            git_client=MockGitClient(diff=_get_large_diff(files=4, lines=20, line_length=200)),
            ai_context_window=8_000,
        )
        code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )

        summarizing_prompt = self._get_prompts(summarizing_agent)[0]
        assert "too large to fit in the context window" in summarizing_prompt


def test_merge_review_responses_keeps_all_comments_and_the_worst_score() -> None:
    first = ReviewResponse(summary="first", raw_score=4, comments=[_get_review_comment("a.py")])
    second = ReviewResponse(summary="second", raw_score=2, comments=[_get_review_comment("b.py")])

    merged = _merge_review_responses([first, second])

    assert merged.summary == "first\n\nsecond"
    assert merged.raw_score == 2
    assert [comment.new_path for comment in merged.comments] == ["a.py", "b.py"]


def test_merge_review_responses_of_a_single_part_is_a_no_op() -> None:
    response = ReviewResponse(summary="only", raw_score=5)

    assert _merge_review_responses([response]) is response


def _get_review_comment(file_path: str) -> ReviewComment:
    return ReviewComment(
        new_path=file_path,
        old_path=file_path,
        line_number=1,
        comment="a comment",
        category="Correctness",
        severity="LOW",
        relative_line_number=1,
        programming_language="python",
        is_comment_on_new_path=True,
    )


def _get_large_diff(*, files: int, lines: int, line_length: int) -> list[DiffResult]:
    return [
        DiffResult(
            metadata=DiffFileMetadata(
                new_file=False,
                deleted_file=False,
                renamed_file=False,
                new_path=f"file-{index}.py",
                old_path=f"file-{index}.py",
            ),
            modified_lines=[
                ModifiedLine(
                    line="x" * line_length,
                    line_number=number,
                    relative_line_number=number,
                    modification_type="added",
                )
                for number in range(1, lines + 1)
            ],
        )
        for index in range(files)
    ]


def _assert_agent_message(
    messages: list[ModelMessage],
    expected_user_prompt: str,
    *,
    expected_messages: int,
    expected_prompts: list[Literal["system-prompt", "user-prompt"]],
) -> None:
    requests = _get_requests_from_messages(messages)
    assert requests

    first_message = requests[0]
    assert len(first_message.parts) == expected_messages
    kinds = [part.part_kind for part in first_message.parts]
    assert kinds == expected_prompts

    user_prompt = next(part for part in first_message.parts if part.part_kind == "user-prompt")
    assert user_prompt.content == expected_user_prompt


def _get_requests_from_messages(messages: list[ModelMessage]) -> list[ModelRequest]:
    return [prompt for prompt in messages if isinstance(prompt, ModelRequest)]
