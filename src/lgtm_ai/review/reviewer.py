import logging
from typing import Final, cast

from lgtm_ai.ai.context_window import estimate_tokens, get_prompt_budget
from lgtm_ai.ai.schemas import (
    AdditionalContext,
    PublishMetadata,
    Review,
    ReviewerDeps,
    ReviewRawScore,
    ReviewResponse,
    SummarizingDeps,
)
from lgtm_ai.base.schemas import LocalRepository, PRUrl
from lgtm_ai.config.handler import ResolvedConfig
from lgtm_ai.git.repository import get_diff_from_local_repo
from lgtm_ai.git_client.base import GitClient
from lgtm_ai.git_client.schemas import IssueContent, PRDiff, PRMetadata
from lgtm_ai.review.chunking import chunk_diffs, render_diff_payload
from lgtm_ai.review.context import ContextRetriever
from lgtm_ai.review.exceptions import (
    ClientUsageLimitsExceededError,
    handle_ai_exceptions,
)
from lgtm_ai.review.prompt_generators import PromptGenerator
from lgtm_ai.review.schemas import PRCodeContext
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

logger = logging.getLogger("lgtm.ai")

MIN_DIFF_BUDGET: Final = 1_000
"""Minimum number of tokens we reserve for the diff of a chunk, no matter how big the rest of the prompt is."""

OMITTED_DIFF_NOTICE: Final = (
    "NOTE: The diff of this PR is too large to fit in the context window of the model, so it is not "
    "included here. Rely on the contents of the review below."
)


class CodeReviewer:
    """
    CodeReviewer orchestrates the automated review of pull requests using AI agents and contextual information.

    This class coordinates the process of reviewing a pull request (PR) by leveraging two pydantic-ai agents:
    - reviewer_agent: Generates the initial review, including comments and scores, based on the PR diff and context.
    - summarizing_agent: Refines and summarizes the initial review for clarity and conciseness.

    Key responsibilities:
    - Retrieve PR metadata and diffs using a GitClient.
    - Gather code context, additional context, and related issue context for the PR using a ContextRetriever.
    - Generate prompts for the AI agents using a PromptGenerator, tailored to the PR and its context.
    - Run the reviewer agent to produce an initial review, including inline comments and a summary.
    - Optionally fetch and incorporate related issue context if configured.
    - Run the summarizing agent to produce a final, polished review response.
    - Track and aggregate usage statistics for AI model calls.
    - Return a Review object containing the PR diff, final review response, and metadata about the review process.

    Main workflow:
        - Fetch PR metadata and diff.
        - Gather code and additional context, and optionally issue context.
        - Generate a review prompt and run the reviewer agent for the initial review.
        - Summarize the initial review using the summarizing agent.
        - Return a Review object with all results and metadata.

    """

    def __init__(
        self,
        *,
        reviewer_agent: Agent[ReviewerDeps, ReviewResponse],
        summarizing_agent: Agent[SummarizingDeps, ReviewResponse],
        model: Model,
        context_retriever: ContextRetriever,
        git_client: GitClient | None,
        config: ResolvedConfig,
    ) -> None:
        """
        Initialize a CodeReviewer instance.

        Args:
            reviewer_agent (Agent[ReviewerDeps, ReviewResponse]):
                AI agent that generates the initial review, including comments and scores, based on the PR diff and context.
            summarizing_agent (Agent[SummarizingDeps, ReviewResponse]):
                AI agent that refines and summarizes the initial review for clarity and conciseness.
            model (Model):
                The AI model to use for both agents (e.g., GPT-4, Claude, etc.).
            context_retriever (ContextRetriever):
                Utility to gather code context, additional context, and related issue context for the PR.
            git_client (GitClient):
                Abstraction for interacting with the git hosting service (GitHub, GitLab, etc.), used to fetch PR metadata and diffs.
            config (ResolvedConfig):
                The resolved configuration object, containing settings for AI limits, context sources, technologies, categories, and more.
        """
        self.reviewer_agent = reviewer_agent
        self.summarizing_agent = summarizing_agent
        self.model = model
        self.git_client = git_client
        self.config = config
        self.context_retriever = context_retriever

    def review(self, target: PRUrl | LocalRepository) -> Review:
        """Perform a full review of the given pull request URL or local git repository and return it."""
        total_usage = RunUsage()
        usage_limits = UsageLimits(input_tokens_limit=self.config.ai_input_tokens_limit)

        if self.git_client and isinstance(target, PRUrl):
            metadata = self.git_client.get_pr_metadata(target)
            pr_diff = self.git_client.get_diff_from_url(target)
        elif isinstance(target, LocalRepository):
            metadata = PRMetadata(title="Local changes with no PR", description="")
            pr_diff = get_diff_from_local_repo(target.repo_path, compare=self.config.compare)
        else:
            raise ValueError("Invalid pr_url type or git_client not configured")

        prompt_generator = PromptGenerator(self.config, metadata)

        initial_review_response = self._perform_initial_review(
            target,
            pr_diff=pr_diff,
            pr_metadata=metadata,
            prompt_generator=prompt_generator,
            total_usage=total_usage,
            usage_limits=usage_limits,
        )
        final_review, final_usage = self._summarize_initial_review(
            pr_diff,
            initial_review_response=initial_review_response,
            prompt_generator=prompt_generator,
            total_usage=total_usage,
            usage_limits=usage_limits,
        )
        logger.info("Final review completed")
        logger.debug(
            "Final review score: %d; Number of comments: %d", final_review.raw_score, len(final_review.comments)
        )

        return Review(
            pr_diff=pr_diff,
            review_response=final_review,
            metadata=PublishMetadata(
                model_name=self.model.model_name, usage=final_usage, config=self.config.model_dump()
            ),
        )

    def _perform_initial_review(
        self,
        pr_url: PRUrl | LocalRepository,
        *,
        pr_diff: PRDiff,
        pr_metadata: PRMetadata,
        prompt_generator: PromptGenerator,
        total_usage: RunUsage,
        usage_limits: UsageLimits,
    ) -> ReviewResponse:
        """Perform an initial review of the PR with the reviewer agent."""
        context = self.context_retriever.get_code_context(target=pr_url, pr_diff=pr_diff)
        additional_context = self.context_retriever.get_additional_context(
            pr_url=pr_url,
            additional_context=self.config.additional_context,
        )
        if self.config.issues_platform and self.config.issues_url and self.config.issues_regex:
            logger.info("Fetching issue context related if possible")
            issue_context = self.context_retriever.get_issues_context(
                issues_url=self.config.issues_url,
                issues_regex=self.config.issues_regex,
                pr_metadata=pr_metadata,
            )
        else:
            issue_context = None

        budget = self._get_prompt_budget()
        review_prompt = prompt_generator.generate_review_prompt(
            pr_diff=pr_diff,
            context=context,
            additional_context=additional_context,
            issue_context=issue_context,
        )

        if estimate_tokens(review_prompt) > budget:
            # The contents of the changed files are the biggest and the most redundant part of the
            # prompt (the diff already contains the changes), so we drop them first.
            logger.warning(
                "The review prompt (~%d tokens) does not fit in the context window of %s (budget is %d tokens); "
                "reviewing without the contents of the changed files",
                estimate_tokens(review_prompt),
                self.model.model_name,
                budget,
            )
            context = PRCodeContext(file_contents=[])
            review_prompt = prompt_generator.generate_review_prompt(
                pr_diff=pr_diff,
                context=context,
                additional_context=additional_context,
                issue_context=issue_context,
            )

        if estimate_tokens(review_prompt) > budget:
            return self._perform_chunked_review(
                pr_diff=pr_diff,
                prompt_generator=prompt_generator,
                additional_context=additional_context,
                issue_context=issue_context,
                budget=budget,
                total_usage=total_usage,
                usage_limits=usage_limits,
            )

        logger.info("Reviewer Agent is performing the initial review")
        return self._run_reviewer_agent(review_prompt, total_usage=total_usage, usage_limits=usage_limits)

    def _perform_chunked_review(
        self,
        *,
        pr_diff: PRDiff,
        prompt_generator: PromptGenerator,
        additional_context: list[AdditionalContext] | None,
        issue_context: IssueContent | None,
        budget: int,
        total_usage: RunUsage,
        usage_limits: UsageLimits,
    ) -> ReviewResponse:
        """Review a PR that is too large for the context window of the model in several parts.

        The diff is split in as many parts as needed, each one is reviewed on its own, and the
        resulting reviews are merged into a single one. It is worse than reviewing everything at
        once (the model sees only a part of the PR at a time), but it is much better than not
        reviewing the PR at all.
        """
        empty_context = PRCodeContext(file_contents=[])
        overhead = estimate_tokens(
            prompt_generator.generate_review_prompt(
                pr_diff=pr_diff,
                context=empty_context,
                additional_context=additional_context,
                issue_context=issue_context,
                diff_payload="",
            )
        )
        if budget - overhead < MIN_DIFF_BUDGET:
            # Whatever is left in the prompt besides the diff is already too big; drop it as well.
            logger.warning("The additional context of the PR does not fit in the context window either; dropping it")
            additional_context, issue_context = None, None
            overhead = estimate_tokens(
                prompt_generator.generate_review_prompt(
                    pr_diff=pr_diff, context=empty_context, additional_context=None, diff_payload=""
                )
            )

        diff_budget = max(budget - overhead, MIN_DIFF_BUDGET)
        chunks = chunk_diffs(prompt_generator.get_reviewable_diffs(pr_diff), budget_tokens=diff_budget)
        logger.warning(
            "The PR is too large to be reviewed in one go; splitting its diff in %d parts of at most %d tokens each",
            len(chunks),
            diff_budget,
        )

        responses: list[ReviewResponse] = []
        for part, chunk in enumerate(chunks, start=1):
            logger.info("Reviewer Agent is reviewing part %d out of %d", part, len(chunks))
            chunk_prompt = prompt_generator.generate_review_prompt(
                pr_diff=pr_diff,
                context=empty_context,
                additional_context=additional_context,
                issue_context=issue_context,
                diff_payload=render_diff_payload(chunk, part=part, total_parts=len(chunks)),
            )
            try:
                responses.append(
                    self._run_reviewer_agent(chunk_prompt, total_usage=total_usage, usage_limits=usage_limits)
                )
            except ClientUsageLimitsExceededError:
                if not responses:
                    raise
                # Publishing a partial review is better than failing after having reviewed most of the PR.
                logger.warning(
                    "Usage limits reached after reviewing %d out of %d parts of the PR; "
                    "the review will only cover those parts. Increase `ai_input_tokens_limit` to review it all",
                    part - 1,
                    len(chunks),
                )
                break

        return _merge_review_responses(responses)

    def _run_reviewer_agent(
        self, review_prompt: str, *, total_usage: RunUsage, usage_limits: UsageLimits
    ) -> ReviewResponse:
        """Run the reviewer agent on the given prompt and return its review."""
        with handle_ai_exceptions():
            raw_res = self.reviewer_agent.run_sync(
                model=self.model,
                user_prompt=review_prompt,
                deps=ReviewerDeps(
                    configured_technologies=self.config.technologies, configured_categories=self.config.categories
                ),
                usage=total_usage,
                usage_limits=usage_limits,
            )
        logger.info("Initial review completed")
        logger.debug(
            "Initial review score: %d; Number of comments: %d", raw_res.output.raw_score, len(raw_res.output.comments)
        )
        # _log_review_output_state("initial", raw_res.output)
        initial_usage = raw_res.usage
        logger.debug(
            f"Initial review usage summary: {initial_usage.requests=} {initial_usage.input_tokens=} {initial_usage.output_tokens=} {initial_usage.total_tokens=}"
        )
        return raw_res.output

    def _get_prompt_budget(self) -> int:
        """Return the maximum size (in tokens) a prompt sent to the model can have."""
        return get_prompt_budget(
            self.model.model_name,
            configured_context_window=self.config.ai_context_window,
            input_tokens_limit=self.config.ai_input_tokens_limit,
        )

    def _summarize_initial_review(
        self,
        pr_diff: PRDiff,
        *,
        initial_review_response: ReviewResponse,
        prompt_generator: PromptGenerator,
        total_usage: RunUsage,
        usage_limits: UsageLimits,
    ) -> tuple[ReviewResponse, RunUsage]:
        """Summarize the initial review with the summarizing agent."""
        logger.info("Summarizing Agent is refining the initial review")
        budget = self._get_prompt_budget()
        summary_prompt = prompt_generator.generate_summarizing_prompt(
            pr_diff=pr_diff, raw_review=initial_review_response
        )
        if estimate_tokens(summary_prompt) > budget:
            logger.warning(
                "The summarizing prompt does not fit in the context window of %s; summarizing without the diff",
                self.model.model_name,
            )
            summary_prompt = prompt_generator.generate_summarizing_prompt(
                pr_diff=pr_diff, raw_review=initial_review_response, diff_payload=OMITTED_DIFF_NOTICE
            )
        if estimate_tokens(summary_prompt) > budget:
            logger.warning(
                "The review itself does not fit in the context window of %s; skipping the summarizing step",
                self.model.model_name,
            )
            return initial_review_response, total_usage

        with handle_ai_exceptions():
            final_res = self.summarizing_agent.run_sync(
                model=self.model,
                user_prompt=summary_prompt,
                deps=SummarizingDeps(configured_categories=self.config.categories),
                usage=total_usage,
                usage_limits=usage_limits,
            )
        usage = final_res.usage
        # _log_review_output_state("final", final_res.output)
        return final_res.output, usage


def _merge_review_responses(responses: list[ReviewResponse]) -> ReviewResponse:
    """Merge the reviews of the several parts of a chunked review into a single one.

    All comments are kept, the summaries are concatenated, and the score of the whole PR is the
    worst score any of its parts got.
    """
    if len(responses) == 1:
        return responses[0]

    summaries = [response.summary.strip() for response in responses if response.summary.strip()]
    return ReviewResponse(
        summary="\n\n".join(summaries),
        comments=[comment for response in responses for comment in response.comments],
        raw_score=cast(ReviewRawScore, min(int(response.raw_score) for response in responses)),
    )


def _log_review_output_state(stage: str, review: ReviewResponse) -> None:
    """Log raw text state of LLM output for debugging formatting issues.

    Checks for escaped sequences (\\n), structural artifacts (<tags>), and other
    anomalies that indicate the LLM produced malformed text fields.
    """
    logger.info("=== REVIEW OUTPUT STATE [%s] ===", stage)

    # Check summary for artifacts
    summary = review.summary
    has_escaped_newlines = "\\n" in summary
    has_structural_tags = any(tag in summary for tag in ["<parameter=", "<comment>", "<summary>"])
    if has_escaped_newlines:
        logger.warning(
            "Review output [%s]: summary contains literal \\n escape sequences (LLM produced escaped text)",
            stage,
        )
    if has_structural_tags:
        logger.warning(
            "Review output [%s]: summary contains structural XML-like tags — LLM may be leaking prompt structure into output",
            stage,
        )

    # Check each comment
    for i, c in enumerate(review.comments):
        text = c.comment
        if "\\n" in text:
            logger.warning(
                "Review output [%s]: comment #%d on %s line %d has literal \\n (%d occurrences) — content preview: %r",
                stage,
                i + 1,
                c.new_path,
                c.line_number,
                text.count("\\n"),
                text[:200],
            )
        if "\\t" in text:
            logger.warning(
                "Review output [%s]: comment #%d on %s line %d has literal \\t",
                stage,
                i + 1,
                c.new_path,
                c.line_number,
            )
        structural = [tag for tag in ["<parameter=", "<comment>", "<summary>"] if tag in text]
        if structural:
            logger.warning(
                "Review output [%s]: comment #%d on %s line %d contains structural tags: %s",
                stage,
                i + 1,
                c.new_path,
                c.line_number,
                structural,
            )

    # Check suggestions
    for i, c in enumerate(review.comments):
        if c.suggestion and c.suggestion.snippet:
            snippet = c.suggestion.snippet
            if "\\n" in snippet:
                logger.warning(
                    "Review output [%s]: suggestion #%d snippet has literal \\n (%d occurrences)",
                    stage,
                    i + 1,
                    snippet.count("\\n"),
                )

    logger.info("=== END REVIEW OUTPUT STATE [%s] ===", stage)
