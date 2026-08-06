from typing import Final

DEFAULT_HTTPX_TIMEOUT: Final[int] = 3

NON_REVIEWABLE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".ico",
        ".tiff",
        ".tif",
        ".svg",
        ".avif",
        ".heic",
        ".heif",
        ".psd",
        # Video and audio
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".mkv",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".m4a",
        # Fonts
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # Compiled artifacts and binaries
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".o",
        ".a",
        ".class",
        ".jar",
        ".war",
        ".wasm",
        # Binary documents
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        # Binary data
        ".db",
        ".sqlite",
        ".sqlite3",
    }
)
"""Extensions whose contents an AI reviewer cannot read.

Fetching these costs two API round trips each — the source branch, then the target branch as a fallback — and always
ends in a decode failure, so they are skipped before any request is made. `.svg` is text, but it is an image: the diff
still shows what changed in it, there is just no point pulling the whole file in as reading context.
"""
