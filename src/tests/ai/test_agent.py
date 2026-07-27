from lgtm_ai.ai.agent import (
    _process_extra_settings,
    get_reviewer_agent_with_settings,
    get_summarizing_agent_with_settings,
)
from lgtm_ai.ai.schemas import AgentSettings


def test_process_extra_settings_empty() -> None:
    assert _process_extra_settings(None) == {}


def test_process_extra_settings_retries_only() -> None:
    assert _process_extra_settings(AgentSettings(retries=3)) == {"retries": 3}


def test_process_extra_settings_maps_request_timeout_to_model_settings() -> None:
    extra = _process_extra_settings(AgentSettings(retries=2, request_timeout=300))

    assert extra == {"retries": 2, "model_settings": {"timeout": 300}}


def test_process_extra_settings_no_timeout_means_no_model_settings() -> None:
    extra = _process_extra_settings(AgentSettings(retries=2, request_timeout=None))

    assert "model_settings" not in extra


def test_agents_get_request_timeout_as_model_settings() -> None:
    settings = AgentSettings(retries=1, request_timeout=120)

    reviewer = get_reviewer_agent_with_settings(settings)
    summarizer = get_summarizing_agent_with_settings(settings)

    assert reviewer.model_settings == {"timeout": 120}
    assert summarizer.model_settings == {"timeout": 120}
