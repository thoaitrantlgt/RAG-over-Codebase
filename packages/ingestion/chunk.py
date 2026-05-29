from typing import Any

from .hashing import content_hash
from .schema import CodeChunk
from .source import line_range, node_text


def create_chunk(
    *,
    repo: str,
    relative_path: str,
    language: str,
    symbol_name: str,
    symbol_kind: str,
    source_bytes: bytes,
    node: Any,
) -> CodeChunk:
    start_line, end_line = line_range(node)
    code_body = node_text(source_bytes, node)
    chunk_id = f"{repo}:{relative_path}:{symbol_name}:{start_line}:{end_line}"

    return CodeChunk(
        repo=repo,
        path=relative_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
        code_body=code_body,
        chunk_id=chunk_id,
        content_hash=content_hash(code_body),
    )
