from unittest import mock

import httpx
import pytest
from lgtm_ai.ai.schemas import AdditionalContext
from lgtm_ai.base.schemas import PRSource, PRUrl
from lgtm_ai.config.constants import DEFAULT_ISSUE_REGEX
from lgtm_ai.git.parser import DiffFileMetadata, DiffResult, ModifiedLine
from lgtm_ai.git_client.base import GitClient
from lgtm_ai.git_client.github import GitHubClient
from lgtm_ai.git_client.gitlab import GitlabClient
from lgtm_ai.git_client.schemas import PRDiff, PRMetadata
from lgtm_ai.review.context import ContextRetriever, IssuesClient, select_files_for_context
from lgtm_ai.review.schemas import PRCodeContext, PRContextFileContents
from tests.review.utils import MockGitClient


class TestAdditionalContext:
    def test_retrieve_additional_context_from_git(self) -> None:
        context_retriever = ContextRetriever(
            git_client=MockGitClient(), issues_client=MockGitClient(), httpx_client=mock.Mock(spec=httpx.Client)
        )

        pr_url = PRUrl(
            full_url="https://example.com/repo/pull/1",
            base_url="https://example.com",
            repo_path="repo",
            pr_number=1,
            source=PRSource.github,
        )

        additional_context = context_retriever.get_additional_context(
            pr_url,
            additional_context=(
                AdditionalContext(prompt="Test context", file_url="relative/path/to/file.txt", context=None),
            ),
        )

        assert additional_context == [
            AdditionalContext(
                file_url="relative/path/to/file.txt",
                prompt="Test context",
                context="contents-of-relative/path/to/file.txt-context",
            )
        ]

    def test_retrieve_additional_context_from_url(self) -> None:
        m_httpx_client = mock.Mock(spec=httpx.Client)
        m_httpx_client.get.return_value = mock.Mock(
            status_code=200,
            text="File contents for https://example.com/file.txt",
        )
        context_retriever = ContextRetriever(
            git_client=MockGitClient(), issues_client=MockGitClient(), httpx_client=m_httpx_client
        )

        pr_url = PRUrl(
            full_url="https://example.com/repo/pull/1",
            base_url="https://example.com",
            repo_path="repo",
            pr_number=1,
            source=PRSource.github,
        )

        additional_context = context_retriever.get_additional_context(
            pr_url,
            additional_context=(
                AdditionalContext(prompt="Test context", file_url="https://example.com/file.txt", context=None),
            ),
        )

        assert additional_context == [
            AdditionalContext(
                file_url="https://example.com/file.txt",
                prompt="Test context",
                context="File contents for https://example.com/file.txt",
            )
        ]

    def test_retrieve_additional_context_from_config(self) -> None:
        context_retriever = ContextRetriever(
            git_client=MockGitClient(), issues_client=MockGitClient(), httpx_client=mock.Mock(spec=httpx.Client)
        )
        pr_url = PRUrl(
            full_url="https://example.com/repo/pull/1",
            base_url="https://example.com",
            repo_path="repo",
            pr_number=1,
            source=PRSource.github,
        )

        additional_context = context_retriever.get_additional_context(
            pr_url,
            additional_context=(
                AdditionalContext(prompt="Test context", file_url=None, context="This is a test context"),
            ),
        )

        assert additional_context == [
            AdditionalContext(file_url=None, prompt="Test context", context="This is a test context")
        ]


@pytest.mark.parametrize("client", [GitlabClient, GitHubClient])
class TestCodeContext:
    def test_get_context_multiple_files(self, client: GitClient) -> None:
        m_client = mock.Mock(spec=client)
        m_client.get_file_contents.side_effect = [
            "lorem ipsum dolor sit amet",
            "surprise",
        ]
        context_retriever = ContextRetriever(
            git_client=m_client, issues_client=MockGitClient(), httpx_client=mock.Mock(spec=httpx.Client)
        )
        pr_diff = PRDiff(
            id=10,
            changed_files=["important.py", "logic.py"],
            target_branch="main",
            source_branch="feature",
            diff=[],
        )

        context = context_retriever.get_code_context(
            PRUrl(
                full_url="https://foo",
                base_url="https://example.com",
                repo_path="path",
                pr_number=1,
                source=PRSource.github,
            ),
            pr_diff=pr_diff,
        )

        assert context == PRCodeContext(
            file_contents=[
                PRContextFileContents(file_path="important.py", content="lorem ipsum dolor sit amet"),
                PRContextFileContents(file_path="logic.py", content="surprise"),
            ]
        )

    def test_get_context_one_file_missing(self, client: GitClient) -> None:
        m_client = mock.Mock(spec=client)
        m_client.get_file_contents.side_effect = [
            None,  # Missing in source branch
            None,  # Missing in target branch
            "surprise",
        ]
        context_retriever = ContextRetriever(
            git_client=m_client, issues_client=MockGitClient(), httpx_client=mock.Mock(spec=httpx.Client)
        )

        pr_diff = PRDiff(
            id=10,
            changed_files=["important.py", "logic.py"],
            target_branch="main",
            source_branch="feature",
            diff=[],
        )

        context = context_retriever.get_code_context(
            PRUrl(
                full_url="https://foo",
                base_url="https://example.com",
                repo_path="path",
                pr_number=1,
                source=PRSource.gitlab,
            ),
            pr_diff=pr_diff,
        )

        assert context == PRCodeContext(
            file_contents=[
                # Notice there is no content for important.py, but the other file is still there
                PRContextFileContents(file_path="logic.py", content="surprise", branch="source"),
            ]
        )


class TestIssueContext:
    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            pytest.param(
                PRMetadata(
                    title="feat(PROJ-123): Implement new feature",
                    description="This PR implements the new feature as described in PROJ-957.",
                ),
                "PROJ-123",
                id="jira-no-hash-in-scope",
            ),
            pytest.param(
                PRMetadata(
                    title="feat(#PROJ-123): Implement new feature",
                    description="This PR implements the new feature as described in PROJ-957.",
                ),
                "PROJ-123",
                id="jira-with-hash-in-scope",
            ),
            pytest.param(
                PRMetadata(
                    title="feat(#123): Implement new feature",
                    description="This PR implements the new feature as described in PROJ-957.",
                ),
                "123",
                id="number-with-hash-in-scope",
            ),
            pytest.param(
                PRMetadata(
                    title="feat(123): Implement new feature",
                    description="This PR implements the new feature as described in PROJ-957.",
                ),
                None,
                id="number-no-hash-in-scope",
            ),
            pytest.param(
                PRMetadata(
                    title="feat(#123,#321): Implement new feature",
                    description="This PR implements the new feature as described in PROJ-957.",
                ),
                None,  # TODO: We may want to support this too!
                id="multiple-issues-in-scope",
            ),
            pytest.param(
                PRMetadata(
                    title="fix: Bug fix for issue #456",
                    description="Closes #456 and relates to PROJ-789.",
                ),
                "456",
                id="number-with-hash-in-title-and-description",
            ),
            pytest.param(
                PRMetadata(
                    title="fix: Bug fix for issue #456",
                    description="Resolves #456 and relates to PROJ-789.",
                ),
                "456",
                id="number-with-resolves-in-description",
            ),
            pytest.param(
                PRMetadata(
                    title="fix: Bug fix for issue 456",
                    description="Does something. refs #456",
                ),
                "456",
                id="number-with-hash-in-description",
            ),
            pytest.param(
                PRMetadata(
                    title="docs: Update documentation",
                    description="No issue reference here.",
                ),
                None,
                id="no-issue-reference",
            ),
            pytest.param(
                PRMetadata(
                    title="chore: Misc updates",
                    description="Refs #1234 in the description.",
                ),
                "1234",
                id="number-with-hash-in-description",
            ),
            pytest.param(
                PRMetadata(
                    title="chore: Misc updates",
                    description="Refs #nothing in the description.",
                ),
                None,
                id="non-numeric-hash-in-description",
            ),
            pytest.param(
                PRMetadata(
                    title="chore: Misc updates",
                    description="Refs 1234 in the description without #.",
                ),
                None,
                id="number-no-hash-in-description",
            ),
        ],
    )
    def test_extract_issue_code_from_metadata(self, metadata: PRMetadata, expected: str | None) -> None:
        context_retriever = ContextRetriever(
            git_client=mock.Mock(spec=GitClient),
            issues_client=mock.Mock(spec=IssuesClient),
            httpx_client=mock.Mock(spec=httpx.Client),
        )

        issue_code = context_retriever._extract_issue_code_from_metadata(metadata, DEFAULT_ISSUE_REGEX)
        assert issue_code == expected


def _diff_entry(path: str, *, with_changes: bool = True) -> DiffResult:
    return DiffResult(
        metadata=DiffFileMetadata(new_file=False, deleted_file=False, renamed_file=False, new_path=path, old_path=path),
        modified_lines=(
            [ModifiedLine(line="x", line_number=1, relative_line_number=1, modification_type="added")]
            if with_changes
            else []
        ),
    )


def _pr_diff(changed_files: list[str], diff: list[DiffResult]) -> PRDiff:
    return PRDiff(id=1, changed_files=changed_files, diff=diff, target_branch="main", source_branch="feature")


class TestSelectFilesForContext:
    def test_keeps_changed_source_files(self) -> None:
        pr_diff = _pr_diff(["a.py", "b.py"], [_diff_entry("a.py"), _diff_entry("b.py")])

        assert select_files_for_context(pr_diff) == ["a.py", "b.py"]

    @pytest.mark.parametrize(
        "file_name",
        ["logo.png", "photo.JPEG", "icon.svg", "clip.mp4", "font.woff2", "bundle.zip", "lib.so", "manual.pdf"],
    )
    def test_drops_files_whose_contents_cannot_be_reviewed(self, file_name: str) -> None:
        """Fetching these costs two API round trips and always ends in a decode failure."""
        pr_diff = _pr_diff(["a.py", file_name], [_diff_entry("a.py"), _diff_entry(file_name)])

        assert select_files_for_context(pr_diff) == ["a.py"]

    def test_drops_files_with_no_modified_lines(self) -> None:
        """A pure rename or a mode-only change has nothing in it for the reviewer to comment on."""
        pr_diff = _pr_diff(["a.py", "renamed.py"], [_diff_entry("a.py"), _diff_entry("renamed.py", with_changes=False)])

        assert select_files_for_context(pr_diff) == ["a.py"]

    def test_keeps_files_the_diff_says_nothing_about(self) -> None:
        """The git clients skip diffs they fail to parse; that is not the same as the file being unchanged."""
        pr_diff = _pr_diff(["a.py", "unparsed.py"], [_diff_entry("a.py")])

        assert select_files_for_context(pr_diff) == ["a.py", "unparsed.py"]

    def test_drops_excluded_files_before_fetching_them(self) -> None:
        """Filtering after the fetch, as the prompt generator does, still pays for the download."""
        pr_diff = _pr_diff(["a.py", "poetry.lock"], [_diff_entry("a.py"), _diff_entry("poetry.lock")])

        assert select_files_for_context(pr_diff, exclude=("poetry.lock",)) == ["a.py"]

    def test_excludes_by_glob(self) -> None:
        pr_diff = _pr_diff(["a.py", "docs/readme.md"], [_diff_entry("a.py"), _diff_entry("docs/readme.md")])

        assert select_files_for_context(pr_diff, exclude=("*.md",)) == ["a.py"]

    def test_no_changed_files(self) -> None:
        assert select_files_for_context(_pr_diff([], [])) == []


class TestCodeContextSkipsUnfetchableFiles:
    def test_images_are_never_requested_from_the_git_client(self) -> None:
        git_client = mock.Mock(spec=GitClient)
        git_client.get_file_contents.return_value = "contents"
        retriever = ContextRetriever(
            git_client=git_client, issues_client=None, httpx_client=mock.Mock(spec=httpx.Client)
        )
        pr_url = PRUrl(
            full_url="https://example.com/repo/pull/1",
            base_url="https://example.com",
            repo_path="repo",
            pr_number=1,
            source=PRSource.github,
        )
        pr_diff = _pr_diff(["a.py", "logo.png"], [_diff_entry("a.py"), _diff_entry("logo.png")])

        context = retriever.get_code_context(pr_url, pr_diff)

        assert [fc.file_path for fc in context.file_contents] == ["a.py"]
        requested = [call.kwargs["file_path"] for call in git_client.get_file_contents.call_args_list]
        assert requested == ["a.py"]
