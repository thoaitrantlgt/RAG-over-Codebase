from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".turbo",
    "__pycache__",
}


def language_for_path(path: Path) -> str | None:
    return SUPPORTED_EXTENSIONS.get(path.suffix)
