from dataclasses import replace

from packages.graph import expand_context
from packages.indexing.dense_store import LocalDenseStore
from packages.indexing.embedding import create_embedding_provider
from packages.indexing.qdrant_store import QdrantDenseStore
from packages.indexing.schema import SearchResult
from packages.indexing.sparse_store import SQLiteSparseStore
from packages.indexing.tantivy_store import TantivySparseStore

from .config import AgentConfig


def retrieve(query: str, config: AgentConfig) -> tuple[list[SearchResult], list[SearchResult]]:
    embedder = create_embedding_provider(
        provider=config.embedding_provider,
        dimensions=config.embedding_dimensions,
        model_name=config.embedding_model,
        cache_dir=config.embedding_cache_dir,
    )
    dense_store = (
        QdrantDenseStore(
            path=config.dense_index,
            collection_name=config.qdrant_collection,
            embeddings=embedder,
        )
        if config.dense_backend == "qdrant"
        else LocalDenseStore(config.dense_index, embedder)
    )
    sparse_store = (
        TantivySparseStore(config.sparse_index)
        if config.sparse_backend == "tantivy"
        else SQLiteSparseStore(config.sparse_index)
    )

    try:
        dense = dense_store.search(query, config.retrieve_top_k)
        sparse = sparse_store.search(query, config.retrieve_top_k)
    finally:
        if hasattr(dense_store, "close"):
            dense_store.close()
        if hasattr(sparse_store, "close"):
            sparse_store.close()

    return dense, sparse


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    *,
    limit: int,
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (1.0 / (k + rank))
            if result.chunk_id not in best or result.score > best[result.chunk_id].score:
                best[result.chunk_id] = result

    fused = [
        replace(best[chunk_id], score=score, source="fused")
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.path, item.start_line))
    return fused[:limit]


def expand_retrieved_context(
    fused: list[SearchResult],
    config: AgentConfig,
) -> list[SearchResult]:
    by_id = {result.chunk_id: result for result in fused}

    for seed in fused:
        expanded = expand_context(
            graph_path=config.graph_path,
            chunk_id=seed.chunk_id,
            max_nodes=config.graph_max_nodes,
            backend=config.graph_backend,
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password,
            neo4j_database=config.neo4j_database,
        )
        for node in expanded["related"]:
            metadata = node.get("metadata", {})
            if "chunk_id" not in metadata:
                continue
            chunk_id = metadata["chunk_id"]
            if chunk_id in by_id:
                continue
            by_id[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                repo=metadata["repo"],
                path=metadata["path"],
                start_line=metadata["start_line"],
                end_line=metadata["end_line"],
                symbol_name=metadata["symbol_name"],
                symbol_kind=metadata["symbol_kind"],
                language=metadata["language"],
                code_body=metadata["code_body"],
                summary=metadata.get("summary", ""),
                score=0.0,
                source="graph",
            )
            if len(by_id) >= config.context_top_k:
                return list(by_id.values())[: config.context_top_k]

    return list(by_id.values())[: config.context_top_k]


def citation(result: SearchResult) -> str:
    return f"{result.repo}/{result.path}:{result.start_line}-{result.end_line}"
