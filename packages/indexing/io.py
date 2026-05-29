import json
from pathlib import Path
from typing import Any, Iterable

from .schema import IndexedChunk


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_indexed_jsonl(path: str | Path, chunks: Iterable[IndexedChunk]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
            handle.write("\n")


def load_indexed_jsonl(path: str | Path) -> list[IndexedChunk]:
    return [IndexedChunk(**record) for record in read_jsonl(path)]
