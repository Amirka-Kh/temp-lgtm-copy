from lgtm_ai.ai.schemas import SupportedAIModels

DEFAULT_AI_MODEL: SupportedAIModels = "gemini-2.5-flash"
DEFAULT_INPUT_TOKEN_LIMIT = 500000
DEFAULT_AI_REQUEST_TIMEOUT = 300
"""Timeout in seconds for each individual request to the AI model. Without it, slow or runaway generations hang until the AI provider client's own timeout (e.g. 600s per attempt for OpenAI-compatible clients)."""
DEFAULT_ISSUE_REGEX = r"(?:refs?|closes?|resolves?)[:\s]*((?:#\d+)|(?:#?[A-Z]+-\d+))|(?:fix|feat|docs|style|refactor|perf|test|build|ci)\((?:#(\d+)|#?([A-Z]+-\d+))\)!?:"
