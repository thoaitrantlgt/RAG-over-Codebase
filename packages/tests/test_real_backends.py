import os
import tempfile
import unittest
from pathlib import Path

from packages.graph import build_graph, expand_context
from packages.indexing.pipeline import index_chunks
from packages.indexing.qdrant_store import QdrantDenseStore
from packages.indexing.tantivy_store import TantivySparseStore
from packages.indexing.embedding import HashingEmbeddingProvider
from packages.ingestion import ingest_repository, write_jsonl


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


class RealBackendTests(unittest.TestCase):
    def test_qdrant_and_tantivy_retrievers_index_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "chunks.jsonl"
            qdrant_path = root / "qdrant"
            tantivy_path = root / "tantivy"

            result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")
            write_jsonl(chunks_path, result.chunks)

            index_chunks(
                chunks_path=chunks_path,
                indexed_chunks_path=root / "indexed_chunks.jsonl",
                dense_index_path=qdrant_path,
                sparse_index_path=tantivy_path,
                summary_cache_path=root / "summary_cache.json",
                dense_backend="qdrant",
                sparse_backend="tantivy",
                qdrant_collection="test_code_chunks",
            )

            dense = QdrantDenseStore(
                path=qdrant_path,
                collection_name="test_code_chunks",
                embeddings=HashingEmbeddingProvider(dimensions=128),
            )
            sparse = TantivySparseStore(tantivy_path)
            try:
                dense_results = dense.search("verify token", top_k=3)
                sparse_results = sparse.search("verify token", top_k=3)
            finally:
                dense.close()

        self.assertEqual(dense_results[0].symbol_name, "AuthService.verify_token")
        self.assertEqual(sparse_results[0].symbol_name, "AuthService.verify_token")
        self.assertEqual(dense_results[0].source, "qdrant")
        self.assertEqual(sparse_results[0].source, "tantivy")

    @unittest.skipUnless(
        os.getenv("NEO4J_URI") and os.getenv("NEO4J_PASSWORD"),
        "Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to run Neo4j integration.",
    )
    def test_neo4j_graph_backend_builds_and_expands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "chunks.jsonl"
            result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")
            write_jsonl(chunks_path, result.chunks)

            graph_result = build_graph(
                repo_path=FIXTURE_REPO,
                repo_name="sample_repo",
                chunks_path=chunks_path,
                graph_path=root / "unused.sqlite",
                backend="neo4j",
                neo4j_uri=os.environ["NEO4J_URI"],
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password=os.environ["NEO4J_PASSWORD"],
                neo4j_database=os.getenv("NEO4J_DATABASE"),
            )
            method = next(
                chunk for chunk in result.chunks if chunk.symbol_name == "AuthService.verify_token"
            )
            expanded = expand_context(
                graph_path=root / "unused.sqlite",
                chunk_id=method.chunk_id,
                backend="neo4j",
                neo4j_uri=os.environ["NEO4J_URI"],
                neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                neo4j_password=os.environ["NEO4J_PASSWORD"],
                neo4j_database=os.getenv("NEO4J_DATABASE"),
            )

        self.assertGreater(graph_result.node_count, 0)
        self.assertEqual(expanded["seed"]["name"], "AuthService.verify_token")


if __name__ == "__main__":
    unittest.main()
