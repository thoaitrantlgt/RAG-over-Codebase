import argparse
import json

from .dense_store import LocalDenseStore
from .embedding import create_embedding_provider
from .pipeline import index_chunks
from .qdrant_store import QdrantDenseStore
from .sparse_store import SQLiteSparseStore
from .tantivy_store import TantivySparseStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query Phase 2 hybrid indexes.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index_parser = subcommands.add_parser("index", help="Index Phase 1 chunks.")
    index_parser.add_argument("--chunks", required=True, help="Input Phase 1 JSONL chunks.")
    index_parser.add_argument(
        "--indexed-chunks",
        default="data/index/indexed_chunks.jsonl",
        help="Output enriched JSONL chunks.",
    )
    index_parser.add_argument(
        "--dense-index",
        default="data/index/dense_index.json",
        help="Output local dense index file.",
    )
    index_parser.add_argument(
        "--sparse-index",
        default="data/index/sparse_index.sqlite",
        help="Output local sparse BM25 index file.",
    )
    index_parser.add_argument(
        "--summary-cache",
        default="data/index/summary_cache.json",
        help="Summary cache path.",
    )
    index_parser.add_argument("--embedding-dimensions", type=int, default=128)
    index_parser.add_argument(
        "--embedding-provider",
        choices=["hashing", "fastembed"],
        default="hashing",
    )
    index_parser.add_argument(
        "--embedding-model",
        default="jinaai/jina-embeddings-v2-base-code",
    )
    index_parser.add_argument("--embedding-cache-dir", default="data/models/fastembed")
    index_parser.add_argument(
        "--dense-backend",
        choices=["local", "qdrant"],
        default="local",
    )
    index_parser.add_argument(
        "--sparse-backend",
        choices=["sqlite", "tantivy"],
        default="sqlite",
    )
    index_parser.add_argument("--qdrant-collection", default="code_chunks")

    search_parser = subcommands.add_parser("search", help="Search local dense and sparse indexes.")
    search_parser.add_argument("--q", required=True, help="Search query.")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--dense-index", default="data/index/dense_index.json")
    search_parser.add_argument("--sparse-index", default="data/index/sparse_index.sqlite")
    search_parser.add_argument("--embedding-dimensions", type=int, default=128)
    search_parser.add_argument(
        "--embedding-provider",
        choices=["hashing", "fastembed"],
        default="hashing",
    )
    search_parser.add_argument(
        "--embedding-model",
        default="jinaai/jina-embeddings-v2-base-code",
    )
    search_parser.add_argument("--embedding-cache-dir", default="data/models/fastembed")
    search_parser.add_argument(
        "--dense-backend",
        choices=["local", "qdrant"],
        default="local",
    )
    search_parser.add_argument(
        "--sparse-backend",
        choices=["sqlite", "tantivy"],
        default="sqlite",
    )
    search_parser.add_argument("--qdrant-collection", default="code_chunks")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "index":
        result = index_chunks(
            chunks_path=args.chunks,
            indexed_chunks_path=args.indexed_chunks,
            dense_index_path=args.dense_index,
            sparse_index_path=args.sparse_index,
            summary_cache_path=args.summary_cache,
            embedding_dimensions=args.embedding_dimensions,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_cache_dir=args.embedding_cache_dir,
            dense_backend=args.dense_backend,
            sparse_backend=args.sparse_backend,
            qdrant_collection=args.qdrant_collection,
        )
        print(f"Indexed chunks: {result.indexed_count}")
        print(f"Indexed chunks JSONL: {result.indexed_chunks_path}")
        print(f"Dense index: {result.dense_index_path}")
        print(f"Sparse index: {result.sparse_index_path}")
        print(f"Summary cache: {result.summary_cache_path}")
        return

    embedding_provider = create_embedding_provider(
        provider=args.embedding_provider,
        dimensions=args.embedding_dimensions,
        model_name=args.embedding_model,
        cache_dir=args.embedding_cache_dir,
    )
    if args.dense_backend == "qdrant":
        dense_store = QdrantDenseStore(
            path=args.dense_index,
            collection_name=args.qdrant_collection,
            embeddings=embedding_provider,
        )
    else:
        dense_store = LocalDenseStore(args.dense_index, embedding_provider)

    sparse_store = (
        TantivySparseStore(args.sparse_index)
        if args.sparse_backend == "tantivy"
        else SQLiteSparseStore(args.sparse_index)
    )
    try:
        payload = {
            "dense": [result.to_dict() for result in dense_store.search(args.q, args.top_k)],
            "sparse": [result.to_dict() for result in sparse_store.search(args.q, args.top_k)],
        }
    finally:
        if hasattr(dense_store, "close"):
            dense_store.close()
        if hasattr(sparse_store, "close"):
            sparse_store.close()

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
