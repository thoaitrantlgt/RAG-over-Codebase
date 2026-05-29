import json
from dataclasses import dataclass
from pathlib import Path

from .extract_python import extract_python_chunks
from .extract_typescript import extract_typescript_chunks
from .parser import parse_source
from .schema import CodeChunk, IngestionError
from .walker import walk_repository


@dataclass(frozen=True)
class IngestionResult:
    chunks: list[CodeChunk]
    errors: list[IngestionError]
    files_scanned: int


def ingest_repository(*, repo_path: str | Path, repo_name: str) -> IngestionResult:
    root_path = Path(repo_path).resolve()
    files = walk_repository(root_path)
    chunks: list[CodeChunk] = []
    errors: list[IngestionError] = []

    for source_file in files:
        relative_path = source_file.path.relative_to(root_path).as_posix()
        try:
            source_bytes = source_file.path.read_bytes()
            parsed = parse_source(source_bytes, source_file.language)
            if parsed.has_error:
                errors.append(
                    IngestionError(
                        path=relative_path,
                        language=source_file.language,
                        error="Tree-sitter parsed the file with syntax errors",
                    )
                )

            extractor = (
                extract_python_chunks
                if source_file.language == "python"
                else extract_typescript_chunks
            )
            chunks.extend(
                extractor(
                    repo=repo_name,
                    relative_path=relative_path,
                    language=source_file.language,
                    source_bytes=source_bytes,
                    root_node=parsed.tree.root_node,
                )
            )
        except Exception as exc:
            errors.append(
                IngestionError(
                    path=relative_path,
                    language=source_file.language,
                    error=str(exc),
                )
            )

    chunks.sort(key=lambda chunk: (chunk.path, chunk.start_line, chunk.symbol_name))
    return IngestionResult(chunks=chunks, errors=errors, files_scanned=len(files))


def write_jsonl(output_path: str | Path, records: list[CodeChunk]) -> None:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            handle.write("\n")
