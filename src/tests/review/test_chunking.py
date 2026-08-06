from lgtm_ai.ai.context_window import estimate_tokens
from lgtm_ai.git.parser import DiffFileMetadata, DiffResult, ModifiedLine
from lgtm_ai.review.chunking import (
    DiffChunk,
    chunk_diffs,
    render_diff_payload,
    serialize_diffs,
    truncate_diff,
)


def _make_diff(file_path: str, *, lines: int = 1, line_length: int = 10) -> DiffResult:
    return DiffResult(
        metadata=DiffFileMetadata(
            new_file=False,
            deleted_file=False,
            renamed_file=False,
            new_path=file_path,
            old_path=file_path,
        ),
        modified_lines=[
            ModifiedLine(
                line="x" * line_length,
                line_number=number,
                relative_line_number=number,
                modification_type="added",
            )
            for number in range(1, lines + 1)
        ],
    )


def test_chunk_diffs_keeps_everything_together_if_it_fits() -> None:
    diffs = [_make_diff("a.py"), _make_diff("b.py")]

    chunks = chunk_diffs(diffs, budget_tokens=100_000)

    assert chunks == [DiffChunk(diffs=diffs, truncated_files=())]


def test_chunk_diffs_splits_diffs_that_do_not_fit() -> None:
    diffs = [_make_diff(f"file-{index}.py", lines=20, line_length=100) for index in range(4)]
    file_tokens = estimate_tokens(serialize_diffs([diffs[0]]))

    chunks = chunk_diffs(diffs, budget_tokens=file_tokens * 2)

    assert [chunk.file_paths for chunk in chunks] == [
        ("file-0.py", "file-1.py"),
        ("file-2.py", "file-3.py"),
    ]
    assert all(estimate_tokens(serialize_diffs(chunk.diffs)) <= file_tokens * 2 for chunk in chunks)


def test_chunk_diffs_never_splits_a_single_file_across_chunks() -> None:
    diffs = [_make_diff("small.py"), _make_diff("big.py", lines=50, line_length=100)]

    chunks = chunk_diffs(diffs, budget_tokens=estimate_tokens(serialize_diffs([diffs[0]])) + 10)

    assert [chunk.file_paths for chunk in chunks] == [("small.py",), ("big.py",)]


def test_chunk_diffs_truncates_files_that_do_not_fit_on_their_own() -> None:
    diffs = [_make_diff("huge.py", lines=100, line_length=100)]

    chunks = chunk_diffs(diffs, budget_tokens=500)

    assert len(chunks) == 1
    assert chunks[0].truncated_files == ("huge.py",)
    assert len(chunks[0].diffs[0].modified_lines) < 100
    assert estimate_tokens(serialize_diffs(chunks[0].diffs)) <= 500


def test_truncate_diff_does_not_touch_diffs_that_fit() -> None:
    diff = _make_diff("a.py", lines=3)

    truncated, was_truncated = truncate_diff(diff, budget_tokens=100_000)

    assert truncated == diff
    assert was_truncated is False


def test_truncate_diff_keeps_at_least_one_line() -> None:
    diff = _make_diff("a.py", lines=3, line_length=10_000)

    truncated, was_truncated = truncate_diff(diff, budget_tokens=100)

    assert was_truncated is True
    assert len(truncated.modified_lines) == 1
    assert truncated.modified_lines[0].line.endswith("[truncated by lgtm]")
    assert len(truncated.modified_lines[0].line) < 10_000


def test_render_diff_payload_of_a_single_chunk_is_just_the_diff() -> None:
    chunk = DiffChunk(diffs=[_make_diff("a.py")])

    assert render_diff_payload(chunk) == serialize_diffs(chunk.diffs)


def test_render_diff_payload_warns_the_model_about_partial_diffs() -> None:
    chunk = DiffChunk(diffs=[_make_diff("a.py")], truncated_files=("a.py",))

    payload = render_diff_payload(chunk, part=2, total_parts=3)

    assert payload.startswith(serialize_diffs(chunk.diffs))
    assert "part 2" in payload
    assert "3 parts" in payload
    assert "truncated" in payload
    assert "a.py" in payload
