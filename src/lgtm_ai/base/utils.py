import fnmatch
import pathlib

from lgtm_ai.base.constants import NON_REVIEWABLE_EXTENSIONS
from lgtm_ai.base.schemas import PRSource


def is_non_reviewable_file(file_name: str) -> bool:
    """Whether the file's contents are of no use as reading context for the AI reviewer."""
    return pathlib.Path(file_name).suffix.lower() in NON_REVIEWABLE_EXTENSIONS


def file_matches_any_pattern(file_name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        full_match = fnmatch.fnmatch(file_name, pattern)
        only_filename_match = fnmatch.fnmatch(pathlib.Path(file_name).name, pattern)
        matches = full_match or only_filename_match
        if matches:
            return True
    return False


def git_source_supports_multiline_suggestions(source: PRSource) -> bool:
    """Check if the given git source supports multiline suggestions.

    GitLab does support specifying suggestions that span multiple lines in single-line comments.
    GitHub requires the comment to be multi-line, and the suggestion does not need special markup with ranges.
    """
    return source == PRSource.gitlab
