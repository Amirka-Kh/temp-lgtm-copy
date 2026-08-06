import json
import textwrap
from typing import Literal
from unittest import mock

import pytest
from lgtm_ai.ai.agent import get_reviewer_agent_with_settings, get_summarizing_agent_with_settings
from lgtm_ai.ai.schemas import AdditionalContext, PublishMetadata, Review, ReviewResponse
from lgtm_ai.base.exceptions import NothingToReviewError
from lgtm_ai.base.schemas import PRSource, PRUrl
from lgtm_ai.config.constants import DEFAULT_AI_MODEL
from lgtm_ai.config.handler import ResolvedConfig
from lgtm_ai.git_client.schemas import ExistingReviewComment, PRDiff
from lgtm_ai.review import CodeReviewer
from lgtm_ai.review.context import ContextRetriever
from lgtm_ai.review.exceptions import (
    ClientUsageLimitsExceededError,
    ContextWindowExceededError,
    InvalidAIResponseError,
    ServerError,
    ServerUsageLimitsExceededError,
    UnknownAIError,
)
from lgtm_ai.review.reviewer import SUMMARIZING_FAILED_WARNING
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


LITELLM_CONTEXT_WINDOW_BODY = {
    "message": (
        "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: "
        "Custom_openaiException - request (101408 tokens) exceeds the available context size (65536 tokens), "
        "try increasing it\nmodel=Qwen3.6-Flash."
    ),
    "type": None,
    "param": None,
    "code": "400",
}
"""Verbatim body of a real failure against a self-hosted Qwen behind litellm."""


def _get_ai_validation_error(*, is_validation_error: bool) -> UnexpectedModelBehavior:
    exc = UnexpectedModelBehavior("Error")
    exc_lvl2 = Exception("Error level 2")
    exc_lvl3 = ValidationError("Error level 3", []) if is_validation_error else Exception("Error level 3")
    exc_lvl2.__context__ = exc_lvl3
    exc.__context__ = exc_lvl2

    return exc


def _get_task_group_wrapped_validation_error() -> UnexpectedModelBehavior:
    """Build the error shape pydantic-ai actually produces: the cause is buried inside an `ExceptionGroup`.

    Agents run inside a task group, so the `ValidationError` is never reachable by following `__context__` alone.
    """
    exc = UnexpectedModelBehavior("Error")
    exc.__context__ = ExceptionGroup("unhandled errors in a TaskGroup", [ValidationError("Invalid", [])])
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

    # The review is embedded verbatim, so it is dumped the same way the prompt generator dumps it.
    expected_review_json = ReviewResponse(summary="a", raw_score=1).model_dump_json(indent=2)
    expected_message = "\n".join(
        [
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
        ]
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
    assert "The categories you should exclusively focus on for your review comments are: Correctness, Quality" in content


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
        (_get_task_group_wrapped_validation_error(), InvalidAIResponseError),
        # Context-window overflow has no status code of its own, so the body is the only signal. It must not be
        # confused with a rate limit, which shares the 413.
        (ModelHTTPError(400, "m", body=LITELLM_CONTEXT_WINDOW_BODY), ContextWindowExceededError),
        (
            ModelHTTPError(400, "m", body={"error": {"message": "This model's maximum context length is 8192 tokens"}}),
            ContextWindowExceededError,
        ),
        (
            ModelHTTPError(413, "m", body={"error": {"message": "prompt is too long: 250000 tokens > 200000 maximum"}}),
            ContextWindowExceededError,
        ),
        (ModelHTTPError(413, "m", body={"error": {"message": "Request too large"}}), ServerUsageLimitsExceededError),
        (ModelHTTPError(400, "m", body={"error": {"message": "Invalid tool schema"}}), UnknownAIError),
        # The model never produced a usable structured output (e.g. it answered with plain text instead of calling
        # the output tool), which is what a model without proper tool-calling support does.
        (UnexpectedModelBehavior("Exceeded maximum output retries (1)"), InvalidAIResponseError),
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


def _review_with_existing_comments(
    existing_comments: list[ExistingReviewComment], *, comments_context: bool = True
) -> list[ModelMessage]:
    """Run a review against a git client carrying `existing_comments`, and return the captured agent messages."""
    git_client = MockGitClient(existing_comments=existing_comments)
    test_agent = get_reviewer_agent_with_settings()
    test_summary_agent = get_summarizing_agent_with_settings()
    with (
        test_agent.override(model=TestModel()),
        test_summary_agent.override(model=TestModel()),
        capture_run_messages() as messages,
    ):
        CodeReviewer(
            reviewer_agent=test_agent,
            summarizing_agent=test_summary_agent,
            model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
            git_client=git_client,
            context_retriever=ContextRetriever(
                git_client=git_client, issues_client=git_client, httpx_client=mock.Mock()
            ),
            config=ResolvedConfig(ai_api_key="", git_api_key="", comments_context=comments_context),
        ).review(target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab))
    return messages


def test_existing_comments_are_sent_to_the_reviewer(context_retriever: ContextRetriever) -> None:
    messages = _review_with_existing_comments(
        [
            ExistingReviewComment(
                body="🦉 The retry counter is never reset", author="lgtm", file_path="file-1.txt", line_number=3
            )
        ]
    )

    prompt = _get_user_prompt(messages)
    assert "EXISTING REVIEW COMMENTS:" in prompt
    assert "The retry counter is never reset" in prompt
    assert "you, on an earlier run" in prompt


def test_existing_comments_outside_the_diff_are_not_sent(context_retriever: ContextRetriever) -> None:
    messages = _review_with_existing_comments(
        [ExistingReviewComment(body="a point on an untouched file", author="dev", file_path="untouched.py")]
    )

    assert "EXISTING REVIEW COMMENTS:" not in _get_user_prompt(messages)


def test_existing_comments_are_not_fetched_when_disabled(context_retriever: ContextRetriever) -> None:
    messages = _review_with_existing_comments(
        [ExistingReviewComment(body="🦉 a previous point", author="lgtm", file_path="file-1.txt")],
        comments_context=False,
    )

    assert "EXISTING REVIEW COMMENTS:" not in _get_user_prompt(messages)


def test_no_existing_comments_section_when_there_are_none(context_retriever: ContextRetriever) -> None:
    assert "EXISTING REVIEW COMMENTS:" not in _get_user_prompt(_review_with_existing_comments([]))


def _get_user_prompt(messages: list[ModelMessage]) -> str:
    requests = _get_requests_from_messages(messages)
    user_prompt = next(part for part in requests[0].parts if part.part_kind == "user-prompt")
    return str(user_prompt.content)


def _get_code_reviewer(
    *,
    reviewer_agent: mock.Mock,
    summarizing_agent: mock.Mock,
    context_retriever: ContextRetriever,
) -> CodeReviewer:
    return CodeReviewer(
        reviewer_agent=reviewer_agent,
        summarizing_agent=summarizing_agent,
        model=mock.Mock(spec=OpenAIChatModel, model_name=DEFAULT_AI_MODEL),
        git_client=MockGitClient(),
        context_retriever=context_retriever,
        config=ResolvedConfig(ai_api_key="", git_api_key=""),
    )


@pytest.mark.parametrize(
    "raised_error",
    [
        UnexpectedModelBehavior("Exceeded maximum output retries (1)"),
        ModelHTTPError(500, "Error"),
        UsageLimitExceeded("Error"),
        AgentRunError("Error"),
    ],
)
def test_initial_review_is_published_when_summarizing_agent_fails(
    context_retriever: ContextRetriever, raised_error: Exception
) -> None:
    """A failing summarizing agent degrades the review to the initial one instead of losing it."""
    initial_review = ReviewResponse(summary="initial summary", raw_score=2)
    reviewer_agent = mock.Mock()
    reviewer_agent.run_sync.return_value = mock.Mock(
        output=initial_review, usage=RunUsage(requests=1, input_tokens=10, output_tokens=5)
    )
    summarizing_agent = mock.Mock()
    summarizing_agent.run_sync.side_effect = raised_error

    code_reviewer = _get_code_reviewer(
        reviewer_agent=reviewer_agent,
        summarizing_agent=summarizing_agent,
        context_retriever=context_retriever,
    )
    review = code_reviewer.review(
        target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
    )

    assert review.review_response == initial_review
    assert review.metadata.warnings == (SUMMARIZING_FAILED_WARNING,)


def test_no_warning_is_added_when_summarizing_agent_succeeds(context_retriever: ContextRetriever) -> None:
    reviewer_agent = mock.Mock()
    reviewer_agent.run_sync.return_value = mock.Mock(
        output=ReviewResponse(summary="initial summary", raw_score=2), usage=RunUsage()
    )
    summarizing_agent = mock.Mock()
    summarizing_agent.run_sync.return_value = mock.Mock(
        output=ReviewResponse(summary="final summary", raw_score=4), usage=RunUsage()
    )

    code_reviewer = _get_code_reviewer(
        reviewer_agent=reviewer_agent,
        summarizing_agent=summarizing_agent,
        context_retriever=context_retriever,
    )
    review = code_reviewer.review(
        target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
    )

    assert review.review_response.summary == "final summary"
    assert review.metadata.warnings == ()


def test_failing_reviewer_agent_still_raises(context_retriever: ContextRetriever) -> None:
    """There is nothing to fall back to when the *initial* review is what failed."""
    reviewer_agent = mock.Mock()
    reviewer_agent.run_sync.side_effect = AgentRunError("Error")

    code_reviewer = _get_code_reviewer(
        reviewer_agent=reviewer_agent,
        summarizing_agent=mock.Mock(),
        context_retriever=context_retriever,
    )

    with pytest.raises(UnknownAIError):
        code_reviewer.review(
            target=PRUrl(full_url="foo", base_url="foo", repo_path="foo", pr_number=1, source=PRSource.gitlab)
        )


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
