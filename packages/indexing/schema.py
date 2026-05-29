from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IndexedChunk:
    repo: str
    path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str
    symbol_kind: str
    code_body: str
    chunk_id: str
    content_hash: str
    summary: str
    indexed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"{self.repo}/{self.path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    repo: str
    path: str
    start_line: int
    end_line: int
    symbol_name: str
    symbol_kind: str
    language: str
    code_body: str
    summary: str
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
