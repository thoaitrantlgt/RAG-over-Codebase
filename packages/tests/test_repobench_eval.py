import tempfile
import unittest
from pathlib import Path

from packages.agent.config import AgentConfig
from packages.evals.metrics import evaluate_ranking
from packages.evals.repobench_r import (
    extract_gold_indices,
    prepare_repobench_records,
    run_repobench_eval,
)
from packages.indexing.pipeline import index_chunks


class RepoBenchEvalTests(unittest.TestCase):
    def test_metrics_handle_multiple_positive_ids(self) -> None:
        metrics = evaluate_ranking(
            ["a", "b", "c", "d"],
            {"c", "x"},
        )

        self.assertEqual(metrics.recall_at_1, 0.0)
        self.assertEqual(metrics.recall_at_5, 0.5)
        self.assertAlmostEqual(metrics.mrr_at_10, 1 / 3)
        self.assertGreater(metrics.ndcg_at_10, 0.0)

    def test_extract_gold_indices_from_binary_labels(self) -> None:
        candidates = [{"code_body": "a"}, {"code_body": "b"}, {"code_body": "c"}]

        indices = extract_gold_indices({"labels": [0, 1, 0]}, candidates)

        self.assertEqual(indices, {1})

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
            )

        self.assertEqual(prep.query_count, 1)
        self.assertEqual(prep.chunk_count, 2)
        self.assertEqual(result.sample_count, 1)
        self.assertEqual(result.metrics.recall_at_10, 1.0)


if __name__ == "__main__":
    unittest.main()
