import json
import tempfile
import unittest
from pathlib import Path

from packages.indexing.dense_store import LocalDenseStore
from packages.indexing.embedding import HashingEmbeddingProvider
from packages.indexing.io import load_indexed_jsonl
from packages.indexing.pipeline import index_chunks
from packages.indexing.sparse_store import SQLiteSparseStore
from packages.ingestion import ingest_repository, write_jsonl


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


class IndexingTests(unittest.TestCase):
    def test_index_chunks_enriches_chunks_and_builds_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._build_indexes(Path(temp_dir))

            indexed = load_indexed_jsonl(paths["indexed_chunks"])
            self.assertEqual(len(indexed), 10)
            self.assertTrue(indexed[0].summary)
            self.assertTrue(paths["dense_index"].exists())
            self.assertTrue(paths["sparse_index"].exists())
            self.assertTrue(paths["summary_cache"].exists())

            cache = json.loads(paths["summary_cache"].read_text(encoding="utf-8"))
            self.assertEqual(len(cache), 10)

    def test_sparse_search_uses_symbol_and_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._build_indexes(Path(temp_dir))
            store = SQLiteSparseStore(paths["sparse_index"])
            try:
                results = store.search("verify_token", top_k=3)
            finally:
                store.close()

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].symbol_name, "AuthService.verify_token")
        self.assertEqual(results[0].source, "sparse")

    def test_dense_search_returns_semantic_like_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._build_indexes(Path(temp_dir))
            store = LocalDenseStore(
                paths["dense_index"],
                HashingEmbeddingProvider(dimensions=128),
            )
            results = store.search("permission access resource", top_k=3)

        self.assertGreaterEqual(len(results), 1)
        self.assertIn("check_permission", [result.symbol_name for result in results])

    def _build_indexes(self, root: Path) -> dict[str, Path]:
        chunks_path = root / "chunks.jsonl"
        indexed_chunks_path = root / "indexed_chunks.jsonl"
        dense_index_path = root / "dense_index.json"
        sparse_index_path = root / "sparse_index.sqlite"
        summary_cache_path = root / "summary_cache.json"

        result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")
        write_jsonl(chunks_path, result.chunks)
        index_result = index_chunks(
            chunks_path=chunks_path,
            indexed_chunks_path=indexed_chunks_path,
            dense_index_path=dense_index_path,
            sparse_index_path=sparse_index_path,
            summary_cache_path=summary_cache_path,
        )

        self.assertEqual(index_result.indexed_count, 10)
        return {
            "chunks": chunks_path,
            "indexed_chunks": indexed_chunks_path,
            "dense_index": dense_index_path,
            "sparse_index": sparse_index_path,
            "summary_cache": summary_cache_path,
        }
