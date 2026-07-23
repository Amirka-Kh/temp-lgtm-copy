import datetime
import json
import logging
import re
import zoneinfo
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Annotated, Final, Literal, Self, get_args
from uuid import uuid4

from lgtm_ai.git_client.schemas import PRDiff

logger = logging.getLogger("lgtm.ai")
from openai.types import ChatModel
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, computed_field, model_validator
from pydantic_ai.models.mistral import LatestMistralModelNames
from pydantic_ai.usage import RunUsage

CommentCategory = Literal["Correctness", "Quality", "Testing", "Security"]
CommentSeverity = Literal["LOW", "MEDIUM", "HIGH"]
CommentSeverityPriority = Literal[1, 2, 3]
ReviewScore = Literal["LGTM", "Nitpicks", "Needs Work", "Needs a Lot of Work", "Abandon"]
ReviewRawScore = (
    Literal[1, 2, 3, 4, 5]
    | Literal[
        "1", "2", "3", "4", "5"
    ]  # TODO(https://github.com/pydantic/pydantic-ai/issues/1691): Gemini returns strings and pydantic-ai errors out when using integers in response models
)
DeepSeekModel = Literal[
    "deepseek-chat",
    "deepseek-reasoner",
]
SupportedGeminiModel = Literal[
    # pydantic-ai does not keep track of all Gemini models available, so we add the ones we want to support explicitly.
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-25",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-25",
]
SupportedAnthopicModel = Literal[
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-latest",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-4-sonnet-20250514",
    "claude-sonnet-4-5",
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-opus-4-1-20250805",
    "claude-opus-4-0",
    "claude-opus-4-20250514",
    "claude-4-opus-20250514",
    "claude-3-opus-latest",
    "claude-3-opus-20240229",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251015",
    "claude-3-haiku-20240307",
]

AnyModel = str
"""Users may use any model name in their local AI server, so we just allow any string."""

SupportedAIModels = (
    ChatModel | SupportedGeminiModel | SupportedAnthopicModel | LatestMistralModelNames | DeepSeekModel | AnyModel
)
"""Type of all supported AI models in lgtm."""

SupportedAIModelsList: Final[tuple[SupportedAIModels, ...]] = (
    get_args(ChatModel)
    + get_args(SupportedGeminiModel)
    + get_args(SupportedAnthopicModel)
    + get_args(LatestMistralModelNames)
    + get_args(DeepSeekModel)
)  # Keep in sync with SupportedAIModels except for AnyModel
"""Tuple of all known supported AI models in lgtm."""


SCORE_MAP: Final[dict[ReviewRawScore, ReviewScore]] = {
    5: "LGTM",
    4: "Nitpicks",
    3: "Needs Work",
    2: "Needs a Lot of Work",
    1: "Abandon",
}

SEVERITY_PRIORITY_MAP: Final[dict[CommentSeverity, CommentSeverityPriority]] = {
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


class CodeSuggestionOffset(BaseModel):
    offset: Annotated[
        int,
        Field(description="Offset relative to the comment line number"),
        AfterValidator(lambda v: abs(v)),
    ]
    direction: Annotated[
        Literal["+", "-", "UP", "DOWN"],  # some LLMs (looking at you gpt5) mess this up and use UP/DOWN instead of +/-.
        Field(description="Direction of the offset. + means below, - means above"),
        AfterValidator(lambda v: v if v in ("+", "-") else ("+" if v == "DOWN" else "-")),
    ]

    @model_validator(mode="after")
    def change_direction_of_zero(self) -> Self:
        # GitLab freaks out if the offset is 0 and the direction is +. We change it to -.
        if self.offset == 0:
            self.direction = "-"
        return self


def _normalize_llm_escapes(value: str) -> str:
    """Normalize escaped sequences and strip structural artifacts from LLM output.

    LLMs may output literal backslash-n (\\n) instead of actual newlines, especially when
    generating structured data via pydantic-ai. They may also leak XML-like structural tags
    into text fields (e.g. <parameter=comments>, <summary>) when confused by malformed input.
    This function normalizes escapes and strips such artifacts.
    """
    if not isinstance(value, str):
        return value

    # Strip structural XML-like tags that LLMs sometimes leak into text fields
    # These appear when the model gets confused by malformed input (e.g. Python dict repr instead of JSON)
    if re.search(r"<\w+[^>]*>", value):
        logger.debug("LLM leaked structural XML-like tags into text field, stripping them")
    result = re.sub(r"<\w+[^>]*>", "", value)
    result = re.sub(r"</\w+>", "", result)

    # Replace literal \n with real newline
    if "\\n" in result:
        logger.debug("LLM output contained literal \\n escape sequences, converting to newlines")
    result = result.replace("\\n", "\n")
    # Replace literal \t with real tab
    if "\\t" in result:
        logger.debug("LLM output contained literal \\t escape sequences, converting to tabs")
    result = result.replace("\\t", "\t")

    return result


def _try_extract_inline_json_array(text: str) -> tuple[str, str | None] | None:
    """Try to extract an inline JSON array containing review comment fields from text.

    Used when the LLM leaks function-calling format — a JSON array with "old_path" or
    "new_path" keys (review comment structure) appears embedded after summary text
    without any keyword separator.

    Uses bracket matching that respects string boundaries so that brackets inside
    JSON string values (e.g. result["text"] in a quote_snippet) don't interfere.

    Example input:
        'PR вносит изменения..., [ { "old_path": ..., "new_path": ... } ]'

    Returns:
        (clean_summary, json_array_str) if a valid comment array is found,
        None if no such pattern exists.
    """
    # Quick pre-check: does the text contain a JSON-like array with comment fields?
    if '[' not in text or '}' not in text:
        logger.debug("No bracket/brace characters found in text, skipping inline JSON extraction")
        return None

    def _find_matching_bracket(opening_pos: int) -> int | None:
        """Find matching ] for [ at opening_pos, respecting strings and escape sequences."""
        depth = 0
        i = opening_pos
        while i < len(text):
            c = text[i]
            # Handle escape sequences (skip paired chars like \\n, \", etc.)
            if c == '\\' and i + 1 < len(text):
                i += 2
                continue
            # Skip everything inside quoted strings — brackets here are data, not structure
            if c == '"':
                i += 1
                while i < len(text):
                    sc = text[i]
                    if sc == '\\' and i + 1 < len(text):
                        i += 2
                        continue
                    if sc == '"':
                        break
                    i += 1
                i += 1  # move past closing quote
                continue
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth < 0:
                    return None  # mismatched brackets (started inside a string)
                if depth == 0:
                    return i
            i += 1
        return None

    def _try_parse_at(bracket_pos: int) -> tuple[str, str] | None:
        """Try to extract and parse JSON at the given bracket position."""
        closing = _find_matching_bracket(bracket_pos)
        if closing is None:
            logger.debug("No matching ']' found for '[' at position %d", bracket_pos)
            return None

        json_candidate = text[bracket_pos:closing + 1]
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, list) and len(parsed) > 0:
                has_comment_fields = any(
                    isinstance(item, dict) and ('old_path' in item or 'new_path' in item)
                    for item in parsed
                )
                if has_comment_fields:
                    logger.debug("Found inline JSON array with %d review comments at position %d", len(parsed), bracket_pos)
                    return text[:bracket_pos].rstrip(), json_candidate
                else:
                    logger.debug("JSON array at position %d found but contains no comment fields", bracket_pos)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Failed to parse JSON candidate at position %d: %s", bracket_pos, exc)
        return None

    # Collect all [ positions from right to left.
    # Prefer [ that's followed by { (array of objects = comments array structure).
    last_bracket = text.rfind('[')

    candidates: list[int] = []
    for pos in range(last_bracket, -1, -1):
        if text[pos] == '[':
            after = text[pos + 1:].lstrip()
            if after and after[0] == '{':
                candidates.append(pos)  # rightmost first (highest priority)

    # Lower-priority: remaining [ positions (still right to left)
    for pos in range(last_bracket, -1, -1):
        if text[pos] == '[' and pos not in candidates:
            candidates.append(pos)

    # Try each candidate in priority order; return first valid match
    for bracket_pos in candidates:
        result = _try_parse_at(bracket_pos)
        if result is not None:
            return result

    return None


def _extract_embedded_comments_from_summary(summary: str) -> tuple[str, str | None]:
    """Extract embedded comment JSON from summary when LLM leaks structure into it.

    When the summarizing agent gets confused, it may output something like:
        "some text about issues\\n<parameter=comments>\\nor comments:\\n[{...}]\\n"

    This function detects such patterns and splits them back into:
        (clean_summary, extracted_json_or_None)

    Also strips trailing artifacts that LLMs sometimes leak after the JSON array,
    such as `` "raw_score": 2`` — these appear when the model outputs a dict-like
    structure with summary, comments, and raw_score keys instead of cleanly
    separating them into response-model fields.

    Returns:
        (clean_summary, extracted_comments_json_or_None)
    """
    if not isinstance(summary, str):
        return summary, None

    logger.debug("Attempting to extract embedded comments from summary (%d chars)", len(summary))

    # Pattern 1: <parameter=comments>...</parameter=comments> or </parameter=content> blocks
    # These are XML-like structural tags the LLM leaks
    param_match = re.search(r'<parameter\s*=\s*"?\s*comments\s*"?[^>]*>(.*?)</\w+>', summary, re.DOTALL | re.IGNORECASE)
    if param_match:
        content = param_match.group(1).strip()
        # Check if it looks like a JSON array start
        if "[" in content and "}" not in content.split("[")[0]:
            logger.debug("Pattern 1 matched: <parameter=comments> block found, extracted %d chars of JSON", len(content))
            return summary[:param_match.start()].rstrip(), content

    # Pattern 2: "or comments:" or ", comments:" followed by JSON array
    # This is the exact pattern we've seen from real LLM output
    comment_keyword = re.search(r'[\n,]\s*(?:or\s+)?comments?\s*:\s*\n', summary, re.IGNORECASE)
    if comment_keyword:
        after = summary[comment_keyword.end():].strip()
        # Try to find a JSON array starting from this position
        bracket_pos = after.find("[")
        if bracket_pos >= 0 and "]" in after:
            json_candidate = after[:after.rfind("]") + 1]
            logger.debug("Pattern 2 matched: keyword separator found, extracted %d chars of JSON", len(json_candidate))
            return summary[:comment_keyword.start()].rstrip(), json_candidate

    # Pattern 3: LLM leaks function-calling format — JSON array with comment structure
    # embedded inline after summary text (no keyword separator). Detect by looking for
    # a JSON array containing "old_path" or "new_path" keys (review comment fields).
    # Example: '...summary text...", [ { "old_path": ..., "new_path": ... } ]'
    candidate = _try_extract_inline_json_array(summary)
    if candidate is not None:
        clean, json_str = candidate
        logger.debug("Pattern 3 matched: found %d chars of embedded JSON at position %d", len(json_str), len(clean))

        # Strip trailing artifacts that LLMs sometimes leak after the JSON array.
        # Real-world output includes "raw_score": N after the closing ] — when the
        # model emits a dict-like structure { "summary": ..., "comments": [...],
        # "raw_score": N } instead of cleanly separating response-model fields.
        # Match: space(s) + "raw_score" + colon + optional whitespace + int/quoted int
        clean = re.sub(r'\s+"raw_score"\s*:\s*(?:["\']?\d+["\']?)\s*$', '', clean).rstrip()

        return candidate

    logger.debug("No embedded comments pattern matched in summary")
    return summary, None


def _parse_comments_json(json_str: str | None) -> list[dict]:
    """Parse a JSON string into a list of comment dicts, returning empty list on failure."""
    if not json_str:
        logger.debug("No JSON string provided to parse_comments_json, returning empty list")
        return []
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            logger.debug("Parsed JSON successfully: top-level array with %d items", len(parsed))
            return parsed
        if isinstance(parsed, dict) and "comments" in parsed:
            comments_data = parsed["comments"]
            if isinstance(comments_data, list):
                logger.debug("Parsed JSON successfully: 'comments' key contains array with %d items", len(comments_data))
                return comments_data
            else:
                logger.debug("JSON 'comments' key exists but is not a list (got %s), returning empty", type(comments_data).__name__)
        else:
            logger.debug("JSON parsed as dict but has no 'comments' key, returning empty")
    except (json.JSONDecodeError, TypeError) as exc:
        logger.debug("Failed to parse JSON: %s. Input was %d chars.", exc, len(json_str))
    return []


def _normalize_llm_escapes_optional(value: str | None) -> str | None:
    """Normalize escaped sequences in optional string fields from LLM output."""
    if value is None:
        return None
    return _normalize_llm_escapes(value)


def _normalize_raw_score(value: ReviewRawScore | str | int) -> ReviewRawScore:
    """Normalize raw_score from LLM output.

    LLMs sometimes double-escape scores, producing '"2"' (a string containing
    literal quote characters around a digit). This strips surrounding quotes
    to recover the intended value so pydantic's Literal validation succeeds.
    """
    if isinstance(value, int):
        return value  # Already an integer — pass through

    if not isinstance(value, str):
        # Non-string, non-int: let Pydantic's normal validation handle it
        return value  # type: ignore[return-value]

    stripped = value.strip()

    # Repeatedly strip matching outer quote pairs (handles '"2"' → '2', "'2'" → '2')
    while len(stripped) >= 2 and (
        (stripped[0] == '"' and stripped[-1] == '"')
        or (stripped[0] == "'" and stripped[-1] == "'")
    ):
        stripped = stripped[1:-1].strip()

    # Validate the result is a valid score digit before returning
    if stripped in ("1", "2", "3", "4", "5"):
        return stripped  # type: ignore[return-value]

    # Pass through for Pydantic to validate — will raise if garbage
    return value  # type: ignore[return-value]


class CodeSuggestion(BaseModel):
    start_offset: Annotated[
        CodeSuggestionOffset,
        Field(description="Offset (from comment line number) to start the suggestion"),
    ]
    end_offset: Annotated[
        CodeSuggestionOffset,
        Field(description="Offset (from comment line number) to end the suggestion"),
    ]
    snippet: Annotated[str, BeforeValidator(_normalize_llm_escapes), Field(description="Suggested code snippet to replace the commented code")]
    programming_language: Annotated[str, Field(description="Programming language of the code snippet")]
    ready_for_replacement: Annotated[
        bool, Field(description="Whether the suggestion is totally ready to be applied directly")
    ] = False


class ReviewComment(BaseModel):
    """Individual comment representation in a PR code review."""

    old_path: Annotated[str, Field(description="Path of the file in the base branch")]
    new_path: Annotated[str, Field(description="Path of the file in the PR branch")]
    comment: Annotated[str, BeforeValidator(_normalize_llm_escapes), Field(description="Review comment")]
    category: Annotated[CommentCategory, Field(description="Category of the comment")]
    severity: Annotated[CommentSeverity, Field(description="Severity of the comment")]
    line_number: Annotated[int, Field(description="Line number to place the comment in the PR")]
    relative_line_number: Annotated[int, Field(description="Relative line number (in the diff) to place the comment")]
    is_comment_on_new_path: Annotated[bool, Field(description="Whether the comment is on a new path")]
    programming_language: Annotated[str, Field(description="Programming language of the file")]
    quote_snippet: Annotated[str | None, BeforeValidator(_normalize_llm_escapes_optional), Field(description="Quoted code snippet")] = None
    suggestion: Annotated[CodeSuggestion | None, Field(description="Suggested code change")] = None


class ReviewResponse(BaseModel):
    """Structured output of any AI agent performing or summarizing code reviews."""

    summary: Annotated[
        str, BeforeValidator(_normalize_llm_escapes), Field(description="Summary of the review")
    ]
    # comments are sorted by severity
    comments: Annotated[
        list[ReviewComment], AfterValidator(lambda v: sorted(v, key=lambda x: SEVERITY_PRIORITY_MAP[x.severity]))
    ] = []
    raw_score: Annotated[
        ReviewRawScore,
        Field(description="Overall score of the review"),
        BeforeValidator(_normalize_raw_score),
        AfterValidator(lambda v: int(v) if isinstance(v, str) else v),
    ]

    @model_validator(mode="after")
    def _extract_embedded_comments(self) -> Self:
        """Extract embedded comment JSON from summary when LLM leaks structure into it.

        When the summarizing agent gets confused, it may embed the comments JSON array
        inside the summary field alongside structural tags like <parameter=comments>.
        This validator detects such patterns and splits them back into proper fields.

        Example bad output:
            summary: "issues found<parameter=comments>or comments:\n[{...}]\n"
            comments: []

        After this validator:
            summary: "issues found"
            comments: [<parsed ReviewComment objects>]
        """
        clean_summary, embedded_json = _extract_embedded_comments_from_summary(self.summary)
        logger.debug("Embedded comments extracted from summary: %s", embedded_json is not None)
        if embedded_json and not self.comments:
            # LLM leaked comment structure into summary — extract it
            raw_comments = _parse_comments_json(embedded_json)
            logger.debug("Raw comments parsed from embedded JSON: %d items", len(raw_comments))
            if raw_comments:
                try:
                    # Convert dict dicts to ReviewComment objects
                    extracted = [ReviewComment.model_validate(rc) for rc in raw_comments]
                    self.comments = extracted
                    self.summary = _normalize_llm_escapes(clean_summary)
                except Exception:
                    # If conversion fails, keep original data — logging will catch it
                    logger.debug(
                        "Failed to parse embedded comments from summary (keeping original data). "
                        "Summary has %d chars, embedded JSON has %d chars",
                        len(self.summary),
                        len(embedded_json) if embedded_json else 0,
                    )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> ReviewScore:
        return SCORE_MAP[self.raw_score]


class GuideKeyChange(BaseModel):
    file_name: Annotated[str, Field(description="File name of the key change")]
    description: Annotated[str, Field(description="Description of the key change")]


class GuideChecklistItem(BaseModel):
    description: Annotated[str, Field(description="Description of the checklist item")]


class GuideReference(BaseModel):
    title: Annotated[str, Field(description="Title of the reference")]
    url: Annotated[str, Field(description="URL of the reference")]


class GuideResponse(BaseModel):
    summary: Annotated[str, Field(description="Summary of the review guide")]
    key_changes: Annotated[list[GuideKeyChange], Field(description="Key changes in the PR")]
    checklist: Annotated[list[GuideChecklistItem], Field(description="Checklist of items to review")]
    references: Annotated[list[GuideReference], Field(description="References to external resources")]


class PublishMetadata(BaseModel):
    model_name: str
    usage: RunUsage
    config: dict[str, object] | None = None

    @cached_property
    def created_at(self) -> str:
        return datetime.datetime.now(zoneinfo.ZoneInfo("UTC")).isoformat()

    @cached_property
    def uuid(self) -> str:
        return uuid4().hex


class Review(BaseModel):
    """Represent a full code review performed by any AI agent."""

    pr_diff: PRDiff
    review_response: ReviewResponse
    metadata: PublishMetadata


class ReviewGuide(BaseModel):
    """Represent a code review guide generated by the AI agent."""

    pr_diff: PRDiff
    guide_response: GuideResponse
    metadata: PublishMetadata


@dataclass(frozen=True, slots=True)
class ReviewerDeps:
    """Dependencies passed to the AI agent performing the code review.

    This is used to generate the system prompt for the AI agent.
    """

    configured_technologies: tuple[str, ...]
    configured_categories: tuple[CommentCategory, ...]


@dataclass(frozen=True, slots=True)
class SummarizingDeps:
    """Dependencies passed to the AI agent summarizing the code review."""

    configured_categories: tuple[CommentCategory, ...]


class AgentSettings(BaseModel):
    """Configurable settings to pass to pydantic-ai agents."""

    retries: Annotated[
        int | None,
        Field(
            description="Number of retries the agent will perform when querying the AI API. Defaults to None, in which case pydantic-ai defaults will be used."
        ),
    ]
    request_timeout: Annotated[
        int | None,
        Field(
            description="Timeout in seconds for each individual request to the AI API. Defaults to None, in which case the AI provider client defaults will be used."
        ),
    ] = None


class AdditionalContext(BaseModel):
    """Additional context for the LLM.

    It is optional and can contain things like project development or stylistic guidelines, common conventions or even LLM-specific instructions. It can come from a file in the project repository, any file on the internet or provided directly.
    """

    file_url: Annotated[
        str | None,
        Field(
            description="Path to a file in the repository or an arbitrary URL to a text file. The context will be read from this resource."
        ),
    ] = None
    prompt: Annotated[
        str,
        Field(
            description="LLM prompt to introduce the context. Can be left empty if the context is descriptive enough."
        ),
    ]
    context: Annotated[
        str | None,
        Field(
            description="Contents of the context itself. Can be provided directly or left empty, to be filled-in from the source pointed to by `file_url`."
        ),
    ] = None
