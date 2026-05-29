import json
from pathlib import Path

from .embedding import EmbeddingProvider, cosine_similarity, embedding_text
from .schema import IndexedChunk, SearchResult


class LocalDenseStore:
    def __init__(self, path: str | Path, embeddings: EmbeddingProvider) -> None:
        self.path = Path(path).resolve()
        self.embeddings = embeddings
        self._records = self._load()

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            vector = self.embeddings.embed(embedding_text(chunk_dict))
            self._records[chunk.chunk_id] = {
                "chunk": chunk_dict,
                "vector": vector,
            }

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._records.pop(chunk_id, None)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_vector = self.embeddings.embed(query)
        scored: list[SearchResult] = []
        for record in self._records.values():
            chunk = record["chunk"]
            score = cosine_similarity(query_vector, record["vector"])
            scored.append(
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
                    score=score,
                    source="dense",
                )
            )

        scored.sort(key=lambda item: (-item.score, item.path, item.start_line))
        return scored[:top_k]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._records, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
