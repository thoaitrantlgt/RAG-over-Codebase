from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolRecord:
    chunk_id: str
    repo: str
    path: str
    symbol_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    metadata: dict[str, Any]


class SymbolResolver:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.records = [SymbolRecord(metadata=chunk, **symbol_record_fields(chunk)) for chunk in chunks]
        self.by_file: dict[str, list[SymbolRecord]] = defaultdict(list)
        self.by_name: dict[str, list[SymbolRecord]] = defaultdict(list)
        for record in self.records:
            self.by_file[record.path].append(record)
            self.by_name[record.symbol_name].append(record)

        for records in self.by_file.values():
            records.sort(key=lambda item: (item.start_line, item.end_line))

    def enclosing_callable(self, path: str, line: int) -> SymbolRecord | None:
        candidates = [
            record
            for record in self.by_file.get(path, [])
            if record.symbol_kind in {"function", "method"}
            and record.start_line <= line <= record.end_line
        ]
        candidates.sort(key=lambda item: (item.end_line - item.start_line, item.symbol_name))
        return candidates[0] if candidates else None

    def resolve_call(self, source_path: str, call_name: str) -> SymbolRecord | None:
        names = [call_name]
        if "." in call_name:
            names.append(call_name.split(".")[-1])

        for name in names:
            same_file = [
                record
                for record in self.by_name.get(name, [])
                if record.path == source_path
            ]
            if same_file:
                return sorted(same_file, key=lambda item: item.start_line)[0]

        suffix = call_name.split(".")[-1]
        suffix_matches = [
            record
            for record in self.records
            if record.symbol_name == suffix or record.symbol_name.endswith(f".{suffix}")
        ]
        same_file_suffix = [record for record in suffix_matches if record.path == source_path]
        if same_file_suffix:
            return sorted(same_file_suffix, key=lambda item: item.start_line)[0]
        if suffix_matches:
            return sorted(suffix_matches, key=lambda item: (item.path, item.start_line))[0]
        return None


def symbol_record_fields(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "repo": chunk["repo"],
        "path": chunk["path"],
        "symbol_name": chunk["symbol_name"],
        "symbol_kind": chunk["symbol_kind"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
    }
