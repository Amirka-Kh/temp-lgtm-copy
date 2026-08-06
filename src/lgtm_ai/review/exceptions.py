import re
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from typing import Final, NoReturn

import openai
from lgtm_ai.base.exceptions import LGTMException
from pydantic import ValidationError
from pydantic_ai import AgentRunError, UnexpectedModelBehavior
from pydantic_ai.exceptions import (
    ModelHTTPError,
    UsageLimitExceeded,
)

_CONTEXT_WINDOW_PATTERN: Final = re.compile(
    r"context[ _]?(window|length|size)|contextwindowexceeded|prompt is too long|too many tokens",
    re.IGNORECASE,
)
"""Wordings providers use when the prompt does not fit the model's context window.

There is no status code for this — litellm and vLLM answer 400, OpenAI answers 400 with `context_length_exceeded`,
and Anthropic answers 413 — so the body is the only reliable signal.
"""

_RETRIES_EXHAUSTED_PATTERN: Final = re.compile(r"Exceeded maximum (output )?retries", re.IGNORECASE)
"""pydantic-ai raises `UnexpectedModelBehavior` with this message when the model never produced a usable structured output.

It covers both the model answering with plain text instead of calling the output tool, and the model calling the
output tool with arguments that do not validate against the response schema.
"""


def _is_caused_by_validation_error(error: BaseException | None, _seen: set[int] | None = None) -> bool:
    """Check whether a `ValidationError` appears anywhere in the causal graph of `error`.

    pydantic-ai runs agents inside a task group, so the original `ValidationError` is usually buried inside an
    `ExceptionGroup` rather than being reachable through `__context__` alone. We therefore walk `__cause__`,
    `__context__` and the members of any exception group, guarding against cycles.
    """
    seen = set() if _seen is None else _seen
    if error is None or id(error) in seen:
        return False
    seen.add(id(error))

    if isinstance(error, ValidationError):
        return True

    candidates: list[BaseException | None] = [error.__cause__, error.__context__]
    if isinstance(error, BaseExceptionGroup):
        candidates.extend(error.exceptions)
    return any(_is_caused_by_validation_error(candidate, seen) for candidate in candidates)


class BaseAIError[T: AgentRunError](LGTMException):
    def __init__(self, message: str = "An error occurred when calling the AI model.") -> None:
        super().__init__(message)

    @classmethod
    def match(cls, error: T) -> bool:
        return False


class ServerUsageLimitsExceededError(BaseAIError[ModelHTTPError]):
    def __init__(
        self, message: str = "The request to the AI model is too large or the AI model has been called too many times."
    ) -> None:
        super().__init__(message)

    @classmethod
    def match(cls, error: ModelHTTPError) -> bool:
        return error.status_code in {HTTPStatus.REQUEST_ENTITY_TOO_LARGE, HTTPStatus.TOO_MANY_REQUESTS}


class ClientUsageLimitsExceededError(BaseAIError[UsageLimitExceeded]):
    def __init__(self, message: str = "The request to the AI model exceeds the configured usage limits.") -> None:
        message += ". You can increase the limit by setting `ai_input_tokens_limit` in the configuration file or through the cli."
        super().__init__(message)

    @classmethod
    def match(cls, error: UsageLimitExceeded) -> bool:
        return True  # There is only one type of UsageLimitExceeded error, so we always match it.


class ContextWindowExceededError(BaseAIError[ModelHTTPError]):
    def __init__(
        self,
        message: str = (
            "The review prompt is larger than the model's context window. "
            "Exclude more files with `exclude`, review a smaller PR, or use a model with a bigger context window. "
            "Note that `ai_input_tokens_limit` does not prevent this: it only counts tokens already spent earlier in "
            "the run, so it cannot catch a first request that is too large on its own."
        ),
    ) -> None:
        super().__init__(message)

    @classmethod
    def match(cls, error: ModelHTTPError) -> bool:
        if error.status_code not in {HTTPStatus.BAD_REQUEST, HTTPStatus.REQUEST_ENTITY_TOO_LARGE}:
            return False
        return bool(_CONTEXT_WINDOW_PATTERN.search(str(error.body)))


class ServerError(BaseAIError[ModelHTTPError]):
    def __init__(self, message: str = "The AI model server is currently unavailable.") -> None:
        super().__init__(message)

    @classmethod
    def match(cls, error: ModelHTTPError) -> bool:
        return error.status_code >= 500 and error.status_code < 600


class UnknownAIError(BaseAIError[AgentRunError]):
    def __init__(self, message: str = "An unknown error occurred when calling the AI model.") -> None:
        super().__init__(message)


class InvalidAIResponseError(BaseAIError[UnexpectedModelBehavior]):
    def __init__(
        self,
        message: str = (
            "The AI model returned an invalid response. Try setting `ai_retries` to a higher value, or using a more powerful LLM. "
            "If you are running the model on a local or OpenAI-compatible server, make sure it supports tool calling: "
            "lgtm asks the model to return its review through a `final_result` tool call, and a model that replies with plain "
            "text instead will always fail here."
        ),
    ) -> None:
        super().__init__(message)

    @classmethod
    def match(cls, error: UnexpectedModelBehavior) -> bool:
        if _RETRIES_EXHAUSTED_PATTERN.search(str(error)):
            return True
        return _is_caused_by_validation_error(error)


class ModelNotFoundError(BaseAIError[ModelHTTPError]):
    def __init__(
        self, message: str = "Cannot find provided AI model, are you sure the name is correct and that it is running?"
    ) -> None:
        super().__init__(message)

    @classmethod
    def match(cls, error: ModelHTTPError) -> bool:
        return error.status_code == HTTPStatus.NOT_FOUND


MAPPED_HTTP_ERRORS: Final[tuple[type[BaseAIError[ModelHTTPError]], ...]] = (
    # Checked before ServerUsageLimitsExceededError: both can be a 413, but a context-window overflow needs a
    # different fix from a rate limit, so the more specific match has to win.
    ContextWindowExceededError,
    ServerUsageLimitsExceededError,
    ServerError,
    ModelNotFoundError,
)
"""HTTP model errors that we know how they look like and can be mapped to a specific error, useful for the user."""

MAPPED_BEHAVIOR_ERRORS: Final[tuple[type[BaseAIError[UnexpectedModelBehavior]], ...]] = (InvalidAIResponseError,)
"""Model behavior errors that we know how they look like and can be mapped to a specific error, useful for the user."""


@contextmanager
def handle_ai_exceptions() -> Iterator[None]:
    """Handle exceptions raised by the AI model in a consistent way.

    We try to give the user as much information as possible, and we use LGTMExceptions.
    That way, users can see tracebacks if they run lgtm in debug mode, but they won't see them in normal mode.
    """

    def _raise_mapped_error[T: AgentRunError](mapped_errors: tuple[type[BaseAIError[T]], ...], err: T) -> NoReturn:
        """Attempt to match a specific error to a tuple of known errors of the same type. Raises an unknown error if no match is found."""
        for mapped_error in mapped_errors:
            if mapped_error.match(err):
                raise mapped_error() from err
        raise UnknownAIError from err

    try:
        yield
    except ModelHTTPError as err:
        _raise_mapped_error(MAPPED_HTTP_ERRORS, err)
    except UnexpectedModelBehavior as err:
        _raise_mapped_error(MAPPED_BEHAVIOR_ERRORS, err)
    except UsageLimitExceeded as err:
        raise ClientUsageLimitsExceededError(err.message) from err
    except AgentRunError as err:
        raise UnknownAIError from err
    except openai.APIConnectionError as err:
        raise ServerError from err
