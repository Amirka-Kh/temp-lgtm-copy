"""Build AI prompts for the discuss command."""

import logging

logger = logging.getLogger("lgtm.discuss")


class DiscussionNote:
    """Represents a single note within a discussion thread."""

    def __init__(self, body: str, author: str, created_at: str, is_position_note: bool = False) -> None:
        self.body = body
        self.author = author
        self.created_at = created_at
        self.is_position_note = is_position_note


class DiscussionThread:
    """Represents a full discussion thread with its notes and code context."""

    def __init__(
        self,
        notes: list[DiscussionNote],
        file_path: str | None = None,
        line_number: int | None = None,
        file_content: str | None = None,
    ) -> None:
        self.notes = notes
        self.file_path = file_path
        self.line_number = line_number
        self.file_content = file_content


def build_discuss_prompt(thread: DiscussionThread, user_message: str) -> str:
    """Build an AI prompt to answer the user's question about a review discussion.

    The prompt includes:
    - The original review note (first note in the thread)
    - Code context snippet (if available)
    - Full conversation history (all notes in the thread)
    - The user's question

    Returns a formatted prompt string ready for the AI model.
    """
    if not thread.notes:
        logger.warning("Discussion thread has no notes")
        return _build_prompt_no_notes(thread, user_message)

    # First note is typically the original review comment
    original_note = thread.notes[0]
    other_notes = thread.notes[1:]

    lines: list[str] = []

    # System-level instruction
    lines.append(
        "You are an AI code reviewer assistant. Your job is to answer questions "
        "about review notes you've posted on a Merge Request/Pull Request."
    )
    lines.append("")

    # Original review note
    lines.append("Here is the original review note I posted:")
    lines.append("```")
    lines.append(original_note.body)
    lines.append("```")
    lines.append("")

    # Full file content (if available) for complete context
    if thread.file_content:
        lines.append("Here is the full source file referenced by the review note:")
        lines.append(f"```{thread.file_path or 'unknown'}")
        lines.append(thread.file_content)
        lines.append("```")
        if thread.line_number:
            lines.append("")
            lines.append(f"The review comment targets line {thread.line_number}.")
            # Show the marked line for quick reference
            file_lines = thread.file_content.splitlines()
            idx = thread.line_number - 1
            if 0 <= idx < len(file_lines):
                lines.append(f"  >>> {file_lines[idx]}")
        lines.append("")

    # Conversation history (existing replies, if any)
    if other_notes:
        lines.append("Here is the conversation history on this thread so far:")
        for note in other_notes:
            lines.append(f"[{note.author}] ({note.created_at}):")
            lines.append("```")
            lines.append(note.body)
            lines.append("```")
            lines.append("")

    # User's question
    lines.append("The user is asking the following question:")
    lines.append("```")
    lines.append(user_message)
    lines.append("```")
    lines.append("")

    # Response instructions
    lines.append(
        "Respond helpfully and concisely to the user's question. "
        "You can reference your original review note, explain your reasoning, "
        "or clarify any misunderstandings. Be professional and constructive."
    )

    return "\n".join(lines)


def _build_prompt_no_notes(thread: DiscussionThread, user_message: str) -> str:
    """Build a fallback prompt when the discussion thread has no notes."""
    lines: list[str] = [
        "You are an AI code reviewer assistant. A user is asking a question about a review discussion.",
        "",
        "The discussion thread appears to be empty or has no retrievable notes.",
        "",
        "The user is asking the following question:",
        "```",
        user_message,
        "```",
        "",
        "If you have code context, use it to provide a helpful response. Otherwise, acknowledge that "
        "the original review note could not be retrieved and answer based on the code context alone.",
    ]

    if thread.file_content:
        lines.insert(6, "")
        lines.insert(6, "Here is the available full source file:")
        lines.insert(6, f"```{thread.file_path or 'unknown'}")
        lines.insert(6, thread.file_content)
        lines.insert(6, "```")

    return "\n".join(lines)
