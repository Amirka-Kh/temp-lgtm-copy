"""Utilities to reason about how much prompt a given AI model can actually take.

lgtm builds prompts that contain the whole diff of a PR plus the contents of every changed
file. For large PRs that easily goes over the context window of the model, which makes the
LLM call fail with an opaque error and, more importantly, leaves the PR without any review.

The helpers here give a cheap (no tokenizer dependency, no extra API calls) estimation of
how big a prompt is and how big it is allowed to be, so callers can shrink the prompt
before sending it instead of failing.
"""

import logging
import math
from typing import Final

logger = logging.getLogger("lgtm.ai")

CHARS_PER_TOKEN: Final = 3.0
"""Conservative characters-per-token ratio used to estimate prompt sizes.

The usual rule of thumb is ~4 characters per token for English prose, but our prompts are
mostly JSON-serialized diffs and source code, which tokenize considerably worse. We use a
lower ratio on purpose: overestimating the size of a prompt is cheap (we shrink it a bit
more than needed), underestimating it is not (the request fails).
"""

DEFAULT_CONTEXT_WINDOW: Final = 128_000
"""Context window assumed for models we do not know about (custom/self-hosted models)."""

PROMPT_BUDGET_RATIO: Final = 0.75
"""Fraction of the context window we allow a single prompt to use.

The rest is left for the system prompt, the tool/output schema and the response itself.
"""

MODEL_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    # Gemini
    "gemini-1.5": 1_000_000,
    "gemini-2.0": 1_000_000,
    "gemini-2.5": 1_000_000,
    "gemini-3": 1_000_000,
    # OpenAI
    "gpt-3.5": 16_385,
    "gpt-4": 8_192,
    "gpt-4-turbo": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-5": 400_000,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
    # Anthropic
    "claude-": 200_000,
    # Mistral
    "mistral": 32_000,
    "mistral-large": 128_000,
    "magistral": 40_000,
    "codestral": 256_000,
    "devstral": 128_000,
    # DeepSeek
    "deepseek": 64_000,
}
"""Known context windows, keyed by model name prefix (longest matching prefix wins).

These are deliberately on the safe side; they only need to be good enough to decide whether
a prompt has to be split, and users can always override them with `ai_context_window`.
"""


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens `text` will take once sent to an LLM."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def get_model_context_window(model_name: str) -> int | None:
    """Return the known context window of the given model, if we know about it."""
    normalized = model_name.lower()
    matching = [prefix for prefix in MODEL_CONTEXT_WINDOWS if normalized.startswith(prefix)]
    if not matching:
        return None
    return MODEL_CONTEXT_WINDOWS[max(matching, key=len)]


def resolve_context_window(model_name: str, configured_context_window: int | None = None) -> int:
    """Resolve the context window to use for the given model.

    An explicitly configured value always wins; otherwise we use the known window of the
    model, falling back to a conservative default for models we do not know about.
    """
    if configured_context_window:
        return configured_context_window

    known = get_model_context_window(model_name)
    if known is None:
        logger.debug(
            "Unknown context window for model %s, assuming %d tokens (set `ai_context_window` to override)",
            model_name,
            DEFAULT_CONTEXT_WINDOW,
        )
        return DEFAULT_CONTEXT_WINDOW
    return known


def get_prompt_budget(
    model_name: str,
    configured_context_window: int | None = None,
    input_tokens_limit: int | None = None,
) -> int:
    """Return the maximum number of tokens a single prompt is allowed to use.

    It is capped by the configured `input_tokens_limit` (if any) so that we do not build
    prompts that are guaranteed to trip the usage limits of the run.
    """
    budget = int(resolve_context_window(model_name, configured_context_window) * PROMPT_BUDGET_RATIO)
    if input_tokens_limit is not None:
        budget = min(budget, input_tokens_limit)
    return budget
