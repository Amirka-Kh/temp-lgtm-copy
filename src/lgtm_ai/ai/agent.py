import logging
from collections.abc import Callable
from typing import Any, Final, TypeGuard, cast, get_args

from lgtm_ai.ai.exceptions import InvalidModelName, MissingAIAPIKey, MissingModelUrl
from lgtm_ai.ai.prompts import GUIDE_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT, SUMMARIZING_SYSTEM_PROMPT
from lgtm_ai.ai.schemas import (
    AgentSettings,
    DeepSeekModel,
    GuideResponse,
    ReviewerDeps,
    ReviewResponse,
    SummarizingDeps,
    SupportedAIModels,
    SupportedAIModelsList,
    SupportedAnthopicModel,
    SupportedGeminiModel,
)
from lgtm_ai.ai.utils import match_model_by_wildcard, select_latest_gemini_model
from openai.types import ChatModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.mistral import LatestMistralModelNames, MistralModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.openai import OpenAIProvider

logger = logging.getLogger("lgtm.ai")


def get_ai_model(model_name: SupportedAIModels | str, api_key: str, model_url: str | None = None) -> Model:  # noqa: C901
    def _is_gemini_model(model_name: SupportedAIModels) -> TypeGuard[SupportedGeminiModel]:
        matched_model = match_model_by_wildcard(model_name, get_args(SupportedGeminiModel))
        return bool(matched_model)

    def _is_openai_model(model_name: SupportedAIModels) -> TypeGuard[ChatModel]:
        return model_name in get_args(ChatModel)

    def _is_anthropic_model(model_name: SupportedAIModels) -> TypeGuard[SupportedAnthopicModel]:
        return model_name in get_args(SupportedAnthopicModel)

    def _is_mistral_model(model_name: SupportedAIModels) -> TypeGuard[LatestMistralModelNames]:
        return model_name in get_args(LatestMistralModelNames)

    def _is_deepseek_model(model_name: SupportedAIModels) -> TypeGuard[DeepSeekModel]:
        return model_name in get_args(DeepSeekModel)

    if model_url:
        logger.info("Using model '%s' via custom OpenAI-compatible endpoint: %s", model_name, model_url)
        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(api_key=api_key, base_url=model_url),
            profile=custom_endpoint_model_profile(model_name),
        )

    if model_name in SupportedAIModelsList and not api_key:
        raise MissingAIAPIKey(model_name=model_name)

    if _is_gemini_model(model_name):
        matches = match_model_by_wildcard(
            model_name,
            cast(tuple[SupportedGeminiModel, ...], get_args(SupportedGeminiModel)),
        )
        if not matches:
            raise InvalidModelName(model_name=model_name)
        return GoogleModel(select_latest_gemini_model(matches), provider=GoogleProvider(api_key=api_key))
    elif _is_openai_model(model_name):
        return OpenAIChatModel(model_name=model_name, provider=OpenAIProvider(api_key=api_key))
    elif _is_anthropic_model(model_name):
        return AnthropicModel(model_name=model_name, provider=AnthropicProvider(api_key=api_key))
    elif _is_mistral_model(model_name):
        return MistralModel(model_name=model_name, provider=MistralProvider(api_key=api_key))
    elif _is_deepseek_model(model_name):
        return OpenAIChatModel(model_name=model_name, provider=DeepSeekProvider(api_key=api_key))
    else:
        # Not known models but no custom URL was provided, so we raise an error
        raise MissingModelUrl(model_name=model_name)


_CUSTOM_ENDPOINT_MODEL_PROFILES: Final[dict[str, Callable[[str], ModelProfile | None]]] = {
    "llama": meta_model_profile,
    "gemma": google_model_profile,
    "qwen": qwen_model_profile,
    "qwq": qwen_model_profile,
    "deepseek": deepseek_model_profile,
    "mistral": mistral_model_profile,
    "gpt-oss": harmony_model_profile,
}
"""Model families commonly self-hosted behind an OpenAI-compatible endpoint, and the profile that makes them work.

`OpenAIProvider` always picks the plain OpenAI profile, which is wrong for these: Qwen, for instance, needs its JSON
schema `$defs` inlined before a local server can turn the `final_result` tool schema into a decoding grammar.
Mirrors what `OllamaProvider.model_profile` does for the same models.
"""


def custom_endpoint_model_profile(model_name: str) -> ModelProfile:
    """Build the model profile to use for a model served from a custom OpenAI-compatible endpoint.

    Starts from the OpenAI-compatible defaults (what `OpenAIProvider` would give us) and layers the known profile of
    the model family on top, if we recognize one. Strict tool definitions are disabled because self-hosted servers
    (llama.cpp, vLLM, Ollama, ...) generally do not implement them.
    """
    lowered = model_name.lower()
    family_profile: ModelProfile | None = None
    for prefix, profile_func in _CUSTOM_ENDPOINT_MODEL_PROFILES.items():
        if lowered.startswith(prefix):
            family_profile = profile_func(lowered)
            logger.debug("Applying '%s' model profile to model '%s'", prefix, model_name)

    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        openai_supports_strict_tool_definition=False,
    ).update(family_profile)


def get_reviewer_agent_with_settings(
    agent_settings: AgentSettings | None = None,
) -> Agent[ReviewerDeps, ReviewResponse]:
    extra_settings = _process_extra_settings(agent_settings)
    agent = Agent(
        deps_type=ReviewerDeps,
        output_type=ReviewResponse,
        **extra_settings,
    )
    agent.system_prompt(get_reviewer_system_prompt)
    return agent


def get_summarizing_agent_with_settings(
    agent_settings: AgentSettings | None = None,
) -> Agent[SummarizingDeps, ReviewResponse]:
    extra_settings = _process_extra_settings(agent_settings)
    agent = Agent(
        deps_type=SummarizingDeps,
        output_type=ReviewResponse,
        **extra_settings,
    )
    agent.system_prompt(get_summarizing_system_prompt)
    return agent


def get_guide_agent_with_settings(
    agent_settings: AgentSettings | None = None,
) -> Agent[None, GuideResponse]:
    extra_settings = _process_extra_settings(agent_settings)
    agent = Agent(
        system_prompt=GUIDE_SYSTEM_PROMPT,
        output_type=GuideResponse,
        **extra_settings,
    )
    return agent


def _process_extra_settings(settings: AgentSettings | None) -> dict[str, Any]:
    """Unpacks extra settings into a dict form.

    These settings have defaults in pydantic-ai and are optional in the API, so we're skipping them completely if they are not set on our side.
    """
    extra_settings: dict[str, Any] = {}
    if settings is not None:
        extra_settings = settings.model_dump(exclude_none=True)
        logger.debug("Extra agent settings: %s", extra_settings)
    return extra_settings


def get_comment_categories(ctx: RunContext[ReviewerDeps]) -> str:
    return f"The categories you should exclusively focus on for your review comments are: {
        ', '.join(ctx.deps.configured_categories)
    }"


def get_reviewer_system_prompt(ctx: RunContext[ReviewerDeps]) -> str:
    return "\n\n".join(
        (
            REVIEWER_SYSTEM_PROMPT.strip(),
            get_pr_technologies(ctx),
            get_comment_categories(ctx),
        )
    )


def get_pr_technologies(ctx: RunContext[ReviewerDeps]) -> str:
    if not ctx.deps.configured_technologies:
        return "You are an expert in whatever technologies the PR is using."
    return f"You are an expert in {', '.join([f'"{tech}"' for tech in ctx.deps.configured_technologies])}."


def get_summarizing_categories(ctx: RunContext[SummarizingDeps]) -> str:
    return f"The only comment categories that you should keep in the review are: {', '.join(ctx.deps.configured_categories)}."


def get_summarizing_system_prompt(ctx: RunContext[SummarizingDeps]) -> str:
    return "\n\n".join(
        (
            SUMMARIZING_SYSTEM_PROMPT.strip(),
            get_summarizing_categories(ctx),
        )
    )
