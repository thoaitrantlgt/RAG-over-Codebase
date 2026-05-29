import json
import sqlite3
from pathlib import Path

from .schema import IndexedChunk, SearchResult


class SQLiteSparseStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        with self.connection:
            for chunk in chunks:
                self.connection.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    (chunk.chunk_id,),
                )
                self.connection.execute(
                    """
                    INSERT INTO chunks_fts (
                        chunk_id,
                        symbol_name,
                        summary,
                        code_body,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.symbol_name,
                        chunk.summary,
                        chunk.code_body,
                        json.dumps(chunk.to_dict(), ensure_ascii=False),
                    ),
                )

    def delete(self, chunk_ids: list[str]) -> None:
        with self.connection:
            for chunk_id in chunk_ids:
                self.connection.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    (chunk_id,),
                )

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        rows = self.connection.execute(
            """
            SELECT metadata, bm25(chunks_fts, 4.0, 2.0, 1.0, 0.0) AS rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (escape_query(query), top_k),
        ).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            chunk = json.loads(row["metadata"])
            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    repo=chunk["repo"],
                    path=chunk["path"],
                    start_line=chunk["start_line"],
                    end_line=chunk["end_line"],
                    symbol_name=chunk["symbol_name"],
                    symbol_kind=chunk["symbol_kind"],
                    language=chunk["language"],
                    code_body=chunk["code_body"],
                    summary=chunk["summary"],
                    score=-float(row["rank"]),
                    source="sparse",
                )
            )
        return results

    def close(self) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                symbol_name,
                summary,
                code_body,
                metadata UNINDEXED
            )
            """
        )
        self.connection.commit()


def escape_query(query: str) -> str:
    terms = [
        term.replace('"', "")
        for term in query.replace(".", " ").replace(":", " ").replace("/", " ").split()
        if term.replace('"', "")
    ]
    if not terms:
        return '""'
    return " OR ".join(f'"{term}"' for term in terms)
