from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .dense_store import LocalDenseStore
from .embedding import create_embedding_provider
from .io import read_jsonl, write_indexed_jsonl
from .qdrant_store import QdrantDenseStore
from .schema import IndexedChunk
from .sparse_store import SQLiteSparseStore
from .summarizer import CachedSummarizer
from .tantivy_store import TantivySparseStore


@dataclass(frozen=True)
class IndexingResult:
    indexed_count: int
    indexed_chunks_path: Path
    dense_index_path: Path
    sparse_index_path: Path
    summary_cache_path: Path


def index_chunks(
    *,
    chunks_path: str | Path,
    indexed_chunks_path: str | Path,
    dense_index_path: str | Path,
    sparse_index_path: str | Path,
    summary_cache_path: str | Path,
    embedding_dimensions: int = 128,
    embedding_provider: str = "hashing",
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code",
    embedding_cache_dir: str | None = None,
    dense_backend: str = "local",
    sparse_backend: str = "sqlite",
    qdrant_collection: str = "code_chunks",
) -> IndexingResult:
    raw_chunks = read_jsonl(chunks_path)
    summarizer = CachedSummarizer(summary_cache_path)
    indexed_at = datetime.now(UTC).isoformat()
    indexed_chunks: list[IndexedChunk] = []

    for chunk in raw_chunks:
        summary = chunk.get("summary") or summarizer.summarize(chunk)
        indexed_chunks.append(
            IndexedChunk(
                repo=chunk["repo"],
                path=chunk["path"],
                language=chunk["language"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                symbol_name=chunk["symbol_name"],
                symbol_kind=chunk["symbol_kind"],
                code_body=chunk["code_body"],
                chunk_id=chunk["chunk_id"],
                content_hash=chunk["content_hash"],
                summary=summary,
                indexed_at=indexed_at,
            )
        )

    write_indexed_jsonl(indexed_chunks_path, indexed_chunks)
    summarizer.save()

    embedder = create_embedding_provider(
        provider=embedding_provider,
        dimensions=embedding_dimensions,
        model_name=embedding_model,
        cache_dir=embedding_cache_dir,
    )
    if dense_backend == "qdrant":
        dense_store = QdrantDenseStore(
            path=dense_index_path,
            collection_name=qdrant_collection,
            embeddings=embedder,
            reset=True,
        )
        try:
            dense_store.upsert(indexed_chunks)
        finally:
            dense_store.close()
    elif dense_backend == "local":
        dense_store = LocalDenseStore(dense_index_path, embedder)
        dense_store.upsert(indexed_chunks)
        dense_store.save()
    else:
        raise ValueError(f"Unsupported dense backend: {dense_backend}")

    if sparse_backend == "tantivy":
        sparse_store = TantivySparseStore(sparse_index_path, reset=True)
        sparse_store.upsert(indexed_chunks)
    elif sparse_backend == "sqlite":
        sparse_store = SQLiteSparseStore(sparse_index_path)
        try:
            sparse_store.upsert(indexed_chunks)
        finally:
            sparse_store.close()
    else:
        raise ValueError(f"Unsupported sparse backend: {sparse_backend}")

    return IndexingResult(
        indexed_count=len(indexed_chunks),
        indexed_chunks_path=Path(indexed_chunks_path).resolve(),
        dense_index_path=Path(dense_index_path).resolve(),
        sparse_index_path=Path(sparse_index_path).resolve(),
        summary_cache_path=Path(summary_cache_path).resolve(),
    )
