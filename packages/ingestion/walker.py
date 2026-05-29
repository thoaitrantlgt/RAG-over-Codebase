from dataclasses import dataclass
from pathlib import Path

from .constants import IGNORED_DIRECTORIES, language_for_path


@dataclass(frozen=True)
class SourceFile:
    path: Path
    language: str


def walk_repository(root_path: str | Path) -> list[SourceFile]:
    root = Path(root_path).resolve()
    files: list[SourceFile] = []

    def visit(directory: Path) -> None:
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.is_dir():
                if entry.name not in IGNORED_DIRECTORIES:
                    visit(entry)
                continue

            if not entry.is_file():
                continue

            language = language_for_path(entry)
            if language:
                files.append(SourceFile(path=entry, language=language))

    visit(root)
    return sorted(files, key=lambda item: item.path.as_posix())
