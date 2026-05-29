import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from .embedding import EmbeddingProvider, embedding_text
from .schema import IndexedChunk, SearchResult


class QdrantDenseStore:
    def __init__(
        self,
        *,
        path: str | Path,
        collection_name: str,
        embeddings: EmbeddingProvider,
        reset: bool = False,
    ) -> None:
        self.path = Path(path).resolve()
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(path=str(self.path))
        self._ensure_collection(reset=reset)

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        points = []
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            points.append(
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector=self.embeddings.embed(embedding_text(chunk_dict)),
                    payload=chunk_dict,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embeddings.embed(query),
            limit=top_k,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            chunk = point.payload or {}
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
                    score=float(point.score),
                    source="qdrant",
                )
            )
        return results

    def close(self) -> None:
        self.client.close()

    def _ensure_collection(self, *, reset: bool) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if reset and exists:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embeddings.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
