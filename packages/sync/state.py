import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.ingestion import CodeChunk


@dataclass(frozen=True)
class SyncState:
    repo: str
    updated_at: str
    chunks: dict[str, dict[str, Any]]
    files: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_state(*, repo_name: str, chunks: list[CodeChunk]) -> SyncState:
    chunk_records: dict[str, dict[str, Any]] = {}
    files: dict[str, list[str]] = {}

    for chunk in chunks:
        chunk_records[chunk.chunk_id] = {
            "path": chunk.path,
            "content_hash": chunk.content_hash,
            "symbol_name": chunk.symbol_name,
            "symbol_kind": chunk.symbol_kind,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }
        files.setdefault(chunk.path, []).append(chunk.chunk_id)

    for path in files:
        files[path].sort()

    return SyncState(
        repo=repo_name,
        updated_at=datetime.now(UTC).isoformat(),
        chunks=chunk_records,
        files=dict(sorted(files.items())),
    )


def load_state(path: str | Path) -> SyncState | None:
    state_path = Path(path)
    if not state_path.exists():
        return None
    data = json.loads(state_path.read_text(encoding="utf-8"))
    return SyncState(
        repo=data["repo"],
        updated_at=data["updated_at"],
        chunks=data.get("chunks", {}),
        files=data.get("files", {}),
    )


def save_state(path: str | Path, state: SyncState) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
