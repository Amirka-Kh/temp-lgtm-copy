from lgtm_ai.ai.prompts import REVIEWER_SYSTEM_PROMPT, SUMMARIZING_SYSTEM_PROMPT


def test_review_prompts_require_russian_human_facing_output() -> None:
    expected_instruction = "Write every human-facing review text in Russian."

    assert expected_instruction in REVIEWER_SYSTEM_PROMPT
    assert expected_instruction in SUMMARIZING_SYSTEM_PROMPT
    assert "translate and rewrite it into natural, professional Russian" in SUMMARIZING_SYSTEM_PROMPT
