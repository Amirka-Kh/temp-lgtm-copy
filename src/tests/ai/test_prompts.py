import pytest
from lgtm_ai.ai.prompts import REVIEWER_SYSTEM_PROMPT, SUMMARIZING_SYSTEM_PROMPT

REVIEW_PROMPTS = (REVIEWER_SYSTEM_PROMPT, SUMMARIZING_SYSTEM_PROMPT)


def test_review_prompts_require_russian_human_facing_output() -> None:
    expected_instruction = "Write every human-facing review text in Russian."

    assert expected_instruction in REVIEWER_SYSTEM_PROMPT
    assert expected_instruction in SUMMARIZING_SYSTEM_PROMPT
    assert "translate and rewrite it into natural, professional Russian" in SUMMARIZING_SYSTEM_PROMPT


@pytest.mark.parametrize("prompt", REVIEW_PROMPTS)
def test_review_prompts_ask_for_the_output_tool(prompt: str) -> None:
    """Both agents produce structured output through the `final_result` tool, so the prompt must ask for exactly that."""
    assert "calling the `final_result` tool" in prompt


@pytest.mark.parametrize("prompt", REVIEW_PROMPTS)
def test_review_prompts_never_ask_for_a_plain_text_or_raw_json_answer(prompt: str) -> None:
    """A model that obeys an instruction to answer in text never calls the output tool, and the whole run fails.

    Weaker models (self-hosted Qwen, for instance) do follow such instructions literally, so the prompts must not
    contain them.
    """
    forbidden = (
        "respond with ONLY a valid JSON object",
        'answer with ONLY "LGTM"',
        "answer exactly with `LGTM`",
    )
    for instruction in forbidden:
        assert instruction not in prompt
