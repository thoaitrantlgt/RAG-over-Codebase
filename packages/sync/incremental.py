import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from packages.ingestion import CodeChunk, ingest_repository

from .git import normalize_changed_files
from .state import SyncState, build_state, load_state, save_state


@dataclass(frozen=True)
class IncrementalSyncResult:
    repo: str
    changed_files: list[str]
    files_added: list[str]
    files_modified: list[str]
    files_deleted: list[str]
    chunks_added: list[str]
    chunks_updated: list[str]
    chunks_deleted: list[str]
    chunks_unchanged: list[str]
    changed_chunks_path: str | None
    full_chunks_path: str | None
    state_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_incremental_sync(
    *,
    repo_path: str | Path,
    repo_name: str,
    state_path: str | Path,
    changed_files: list[str] | None = None,
    changed_chunks_out: str | Path | None = None,
    full_chunks_out: str | Path | None = None,
) -> IncrementalSyncResult:
    previous_state = load_state(state_path)
    ingestion = ingest_repository(repo_path=repo_path, repo_name=repo_name)
    current_chunks = ingestion.chunks
    current_state = build_state(repo_name=repo_name, chunks=current_chunks)

    if changed_files is None or not changed_files:
        changed = infer_changed_files(previous_state, current_state)
    else:
        changed = normalize_changed_files(changed_files)

    diff = diff_states(previous_state, current_state, changed)
    changed_chunk_ids = set(diff["chunks_added"]) | set(diff["chunks_updated"])
    changed_chunk_records = [
        chunk for chunk in current_chunks if chunk.chunk_id in changed_chunk_ids or chunk.path in changed
    ]

    if changed_chunks_out is not None:
        write_chunks_jsonl(changed_chunks_out, changed_chunk_records)
    if full_chunks_out is not None:
        write_chunks_jsonl(full_chunks_out, current_chunks)

    save_state(state_path, current_state)

    return IncrementalSyncResult(
        repo=repo_name,
        changed_files=changed,
        files_added=diff["files_added"],
        files_modified=diff["files_modified"],
        files_deleted=diff["files_deleted"],
        chunks_added=diff["chunks_added"],
        chunks_updated=diff["chunks_updated"],
        chunks_deleted=diff["chunks_deleted"],
        chunks_unchanged=diff["chunks_unchanged"],
        changed_chunks_path=str(Path(changed_chunks_out).resolve()) if changed_chunks_out else None,
        full_chunks_path=str(Path(full_chunks_out).resolve()) if full_chunks_out else None,
        state_path=str(Path(state_path).resolve()),
    )


def infer_changed_files(previous: SyncState | None, current: SyncState) -> list[str]:
    if previous is None:
        return sorted(current.files)

    paths = set(previous.files) | set(current.files)
    changed = []
    for path in paths:
        previous_ids = set(previous.files.get(path, []))
        current_ids = set(current.files.get(path, []))
        if previous_ids != current_ids:
            changed.append(path)
            continue
        for chunk_id in previous_ids:
            previous_hash = previous.chunks.get(chunk_id, {}).get("content_hash")
            current_hash = current.chunks.get(chunk_id, {}).get("content_hash")
            if previous_hash != current_hash:
                changed.append(path)
                break
    return sorted(changed)


def diff_states(
    previous: SyncState | None,
    current: SyncState,
    changed_files: list[str],
) -> dict[str, list[str]]:
    previous_files = previous.files if previous else {}
    previous_chunks = previous.chunks if previous else {}

    files_added: list[str] = []
    files_modified: list[str] = []
    files_deleted: list[str] = []
    chunks_added: list[str] = []
    chunks_updated: list[str] = []
    chunks_deleted: list[str] = []
    chunks_unchanged: list[str] = []

    for path in changed_files:
        previous_ids = set(previous_files.get(path, []))
        current_ids = set(current.files.get(path, []))

        if not previous_ids and current_ids:
            files_added.append(path)
        elif previous_ids and not current_ids:
            files_deleted.append(path)
        elif previous_ids != current_ids:
            files_modified.append(path)
        elif previous_ids:
            files_modified.append(path)

        chunks_added.extend(sorted(current_ids - previous_ids))
        chunks_deleted.extend(sorted(previous_ids - current_ids))

        for chunk_id in sorted(previous_ids & current_ids):
            previous_hash = previous_chunks.get(chunk_id, {}).get("content_hash")
            current_hash = current.chunks.get(chunk_id, {}).get("content_hash")
            if previous_hash == current_hash:
                chunks_unchanged.append(chunk_id)
            else:
                chunks_updated.append(chunk_id)

    return {
        "files_added": sorted(set(files_added)),
        "files_modified": sorted(set(files_modified) - set(files_added) - set(files_deleted)),
        "files_deleted": sorted(set(files_deleted)),
        "chunks_added": sorted(set(chunks_added)),
        "chunks_updated": sorted(set(chunks_updated)),
        "chunks_deleted": sorted(set(chunks_deleted)),
        "chunks_unchanged": sorted(set(chunks_unchanged)),
    }


def write_chunks_jsonl(path: str | Path, chunks: list[CodeChunk]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
            handle.write("\n")
