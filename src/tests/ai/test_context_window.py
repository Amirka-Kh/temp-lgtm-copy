import pytest
from lgtm_ai.ai.context_window import (
    DEFAULT_CONTEXT_WINDOW,
    PROMPT_BUDGET_RATIO,
    estimate_tokens,
    get_model_context_window,
    get_prompt_budget,
    resolve_context_window,
)


def test_estimate_tokens_is_conservative() -> None:
    # We deliberately estimate more tokens than the usual ~4 chars per token rule of thumb
    assert estimate_tokens("a" * 100) > 25


def test_estimate_tokens_of_empty_string() -> None:
    assert estimate_tokens("") == 0


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("gemini-2.5-pro", 1_000_000),
        ("gpt-4o", 128_000),
        ("gpt-4", 8_192),  # the longest matching prefix wins, so gpt-4 does not get gpt-4o's window
        ("claude-sonnet-4-5", 200_000),
        ("deepseek-chat", 64_000),
        ("GPT-4O-MINI", 128_000),  # model names are matched case-insensitively
        ("my-self-hosted-model", None),
    ],
)
def test_get_model_context_window(model_name: str, expected: int | None) -> None:
    assert get_model_context_window(model_name) == expected


def test_resolve_context_window_falls_back_to_default_for_unknown_models() -> None:
    assert resolve_context_window("my-self-hosted-model") == DEFAULT_CONTEXT_WINDOW


def test_resolve_context_window_prefers_the_configured_value() -> None:
    assert resolve_context_window("gemini-2.5-pro", configured_context_window=1_234) == 1_234


def test_get_prompt_budget_leaves_room_for_the_response() -> None:
    assert get_prompt_budget("gpt-4o") == int(128_000 * PROMPT_BUDGET_RATIO)


def test_get_prompt_budget_is_capped_by_the_input_tokens_limit() -> None:
    assert get_prompt_budget("gemini-2.5-pro", input_tokens_limit=10_000) == 10_000


def test_get_prompt_budget_ignores_input_tokens_limits_larger_than_the_context_window() -> None:
    assert get_prompt_budget("gpt-4o", input_tokens_limit=1_000_000) == int(128_000 * PROMPT_BUDGET_RATIO)
