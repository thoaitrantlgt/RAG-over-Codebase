import tempfile
import unittest
from pathlib import Path

from packages.agent.config import AgentConfig
from packages.evals.metrics import evaluate_ranking
from packages.evals.repobench_r import (
    build_query,
    extract_gold_indices,
    extract_symbol_name,
    prepare_repobench_records,
    run_repobench_eval,
)
from packages.evals.rerank import bge_m3_rerank, extract_import_names, lexical_rerank
from packages.indexing.pipeline import index_chunks
from packages.indexing.schema import SearchResult


class RepoBenchEvalTests(unittest.TestCase):
    def test_metrics_handle_multiple_positive_ids(self) -> None:
        metrics = evaluate_ranking(
            ["a", "b", "c", "d"],
            {"c", "x"},
        )

        self.assertEqual(metrics.recall_at_1, 0.0)
        self.assertEqual(metrics.recall_at_5, 0.5)
        self.assertAlmostEqual(metrics.mrr_at_10, 1 / 3)
        self.assertEqual(metrics.recall_at_50, 0.5)
        self.assertGreater(metrics.ndcg_at_10, 0.0)

    def test_extract_gold_indices_from_binary_labels(self) -> None:
        candidates = [{"code_body": "a"}, {"code_body": "b"}, {"code_body": "c"}]

        indices = extract_gold_indices({"labels": [0, 1, 0]}, candidates)

        self.assertEqual(indices, {1})

    def test_prepare_extracts_real_symbol_names(self) -> None:
        self.assertEqual(extract_symbol_name("def verify_token(token):\n    pass"), "verify_token")
        self.assertEqual(extract_symbol_name("class AuthService:\n    pass"), "AuthService")

    def test_build_query_keeps_imports_when_code_is_long(self) -> None:
        query = build_query(
            {
                "repo_name": "sample",
                "import_statement": "from auth import verify_token",
                "code": "x = 1\n" * 2000,
            },
            max_chars=1000,
        )

        self.assertIn("from auth import verify_token", query)
        self.assertLessEqual(len(query), 1100)

    def test_lexical_rerank_uses_import_names(self) -> None:
        results = [
            search_result("a", "charge_card", "def charge_card(amount): pass"),
            search_result("b", "verify_token", "def verify_token(token): pass"),
        ]

        reranked = lexical_rerank("from auth import verify_token\ncode:\nvalue = verify_", results)

        self.assertEqual(reranked[0].chunk_id, "b")
        self.assertEqual(extract_import_names("from auth import verify_token"), {"verify_token"})

    def test_bge_rerank_uses_model_scores(self) -> None:
        results = [
            search_result("a", "first", "def first(): pass"),
            search_result("b", "second", "def second(): pass"),
        ]

        reranked = bge_m3_rerank(
            "query",
            results,
            reranker=FakeReranker([0.1, 0.9]),
        )

        self.assertEqual([item.chunk_id for item in reranked], ["b", "a"])
        self.assertEqual(reranked[0].source, "test+bge_m3_rerank")

    def test_prepare_and_run_eval_with_local_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "chunks.jsonl"
            queries_path = root / "queries.jsonl"
            records = [
                {
                    "id": "q1",
                    "repo_name": "sample",
                    "file_path": "src/api.py",
                    "query": "How does the service verify jwt auth tokens?",
                    "candidate_snippets": [
                        {
                            "path": "src/payments.py",
                            "code": "def charge_card(amount):\n    return gateway.charge(amount)",
                        },
                        {
                            "path": "src/auth.py",
                            "code": "def verify_token(token):\n    return jwt.decode(token, key)",
                        },
                    ],
                    "gold_snippet_index": 1,
                }
            ]

            prep = prepare_repobench_records(
                records,
                chunks_path=chunks_path,
                queries_path=queries_path,
            )
            index_chunks(
                chunks_path=chunks_path,
                indexed_chunks_path=root / "indexed.jsonl",
                dense_index_path=root / "dense.json",
                sparse_index_path=root / "sparse.sqlite",
                summary_cache_path=root / "summary.json",
            )
            config = AgentConfig(
                dense_backend="local",
                dense_index=str(root / "dense.json"),
                sparse_backend="sqlite",
                sparse_index=str(root / "sparse.sqlite"),
                embedding_provider="hashing",
                retrieve_top_k=10,
                fused_top_k=10,
            )
            result = run_repobench_eval(
                queries_path=queries_path,
                config=config,
                candidate_top_k=10,
            )

        self.assertEqual(prep.query_count, 1)
        self.assertEqual(prep.chunk_count, 2)
        self.assertEqual(result.sample_count, 1)
        self.assertEqual(result.metrics.recall_at_10, 1.0)

    def test_sample_scope_filters_cross_sample_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "chunks.jsonl"
            queries_path = root / "queries.jsonl"
            records = [
                {
                    "id": "q1",
                    "query": "common helper",
                    "context": ["def common_helper(): pass", "def rare_auth_token(): pass"],
                    "gold_snippet_index": 1,
                },
                {
                    "id": "q2",
                    "query": "common helper",
                    "context": ["def common_helper(): pass", "def another_helper(): pass"],
                    "gold_snippet_index": 0,
                },
            ]
            prepare_repobench_records(
                records,
                chunks_path=chunks_path,
                queries_path=queries_path,
            )
            index_chunks(
                chunks_path=chunks_path,
                indexed_chunks_path=root / "indexed.jsonl",
                dense_index_path=root / "dense.json",
                sparse_index_path=root / "sparse.sqlite",
                summary_cache_path=root / "summary.json",
            )
            config = AgentConfig(
                dense_backend="local",
                dense_index=str(root / "dense.json"),
                sparse_backend="sqlite",
                sparse_index=str(root / "sparse.sqlite"),
                embedding_provider="hashing",
                retrieve_top_k=10,
                fused_top_k=10,
            )
            result = run_repobench_eval(
                queries_path=queries_path,
                config=config,
                scope="sample",
                candidate_top_k=10,
            )

        for item in result.per_query:
            prefix = f"repobench-r:{item['query_id']}:candidate:"
            self.assertTrue(all(chunk_id.startswith(prefix) for chunk_id in item["ranked_chunk_ids"]))

def search_result(chunk_id: str, symbol_name: str, code_body: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        repo="repo",
        path=f"{chunk_id}.py",
        start_line=1,
        end_line=1,
        symbol_name=symbol_name,
        symbol_kind="function",
        language="python",
        code_body=code_body,
        summary="",
        score=0.0,
        source="test",
    )


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def compute_score(self, pairs, batch_size=8, normalize=True):
        return self.scores


if __name__ == "__main__":
    unittest.main()
