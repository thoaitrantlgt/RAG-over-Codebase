from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CodeChunk:
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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class IngestionError:
    path: str
    language: str
    error: str

    def to_dict(self) -> dict:
        return asdict(self)
