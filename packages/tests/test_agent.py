import tempfile
import unittest
from pathlib import Path

from packages.agent.config import AgentConfig
from packages.agent.lmstudio import select_prompt_context
from packages.agent.retrieval import reciprocal_rank_fusion
from packages.agent.workflow import run_agent
from packages.graph import build_graph
from packages.indexing.pipeline import index_chunks
from packages.indexing.schema import SearchResult
from packages.ingestion import ingest_repository, write_jsonl


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


class FakeSynthesizer:
    def synthesize(self, query, context):
        return f"fake answer for {query}: {context[0].symbol_name}"


class AgentTests(unittest.TestCase):
    def test_rrf_deduplicates_and_prefers_items_seen_in_both_lists(self) -> None:
        shared_dense = result("shared", 0.9, "dense")
        dense_only = result("dense-only", 0.8, "dense")
        shared_sparse = result("shared", 12.0, "sparse")
        sparse_only = result("sparse-only", 11.0, "sparse")

        fused = reciprocal_rank_fusion(
            [[shared_dense, dense_only], [shared_sparse, sparse_only]],
            limit=3,
            k=60,
        )

        self.assertEqual(fused[0].chunk_id, "shared")
        self.assertEqual(len({item.chunk_id for item in fused}), 3)
        self.assertEqual(fused[0].source, "fused")

    def test_run_agent_can_skip_lmstudio_and_return_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_agent_fixture(Path(temp_dir))
            answer = run_agent("verify token", config, synthesize=False)

        self.assertIn("Synthesis skipped", answer.answer)
        self.assertGreaterEqual(len(answer.context), 1)
        self.assertIn("AuthService.verify_token", [item.symbol_name for item in answer.context])

    def test_run_agent_uses_injected_synthesizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_agent_fixture(Path(temp_dir))
            answer = run_agent(
                "verify token",
                config,
                synthesizer=FakeSynthesizer(),
            )

        self.assertIn("fake answer", answer.answer)
        self.assertGreaterEqual(len(answer.citations), 1)

    def test_prompt_context_prefers_exact_symbol_and_skips_huge_noise(self) -> None:
        exact = result("solve", 1.0, "fused")
        exact = SearchResult(
            **{
                **exact.__dict__,
                "symbol_name": "solve_dependencies",
                "path": "fastapi/dependencies/utils.py",
            }
        )
        huge_class = result("fastapi-class", 0.9, "fused")
        huge_class = SearchResult(
            **{
                **huge_class.__dict__,
                "symbol_name": "FastAPI",
                "symbol_kind": "class",
                "code_body": "class FastAPI:\n" + ("    pass\n" * 5000),
            }
        )
        small_related = result("related", 0.8, "graph")

        selected = select_prompt_context(
            "solve_dependencies trong FastAPI làm gì?",
            [exact, huge_class, small_related],
            limit=2,
        )

        self.assertEqual([item.symbol_name for item in selected], ["solve_dependencies", "related"])

    def _build_agent_fixture(self, root: Path) -> AgentConfig:
        chunks_path = root / "chunks.jsonl"
        graph_path = root / "graph.sqlite"
        result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")
        write_jsonl(chunks_path, result.chunks)

        index_chunks(
            chunks_path=chunks_path,
            indexed_chunks_path=root / "indexed.jsonl",
            dense_index_path=root / "dense.json",
            sparse_index_path=root / "sparse.sqlite",
            summary_cache_path=root / "summary.json",
        )
        build_graph(
            repo_path=FIXTURE_REPO,
            repo_name="sample_repo",
            chunks_path=chunks_path,
            graph_path=graph_path,
        )

        return AgentConfig(
            dense_backend="local",
            dense_index=str(root / "dense.json"),
            sparse_backend="sqlite",
            sparse_index=str(root / "sparse.sqlite"),
            embedding_provider="hashing",
            graph_backend="sqlite",
            graph_path=str(graph_path),
            retrieve_top_k=5,
            fused_top_k=3,
            context_top_k=5,
        )


def result(chunk_id: str, score: float, source: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        repo="repo",
        path=f"{chunk_id}.py",
        start_line=1,
        end_line=2,
        symbol_name=chunk_id,
        symbol_kind="function",
        language="python",
        code_body="def x(): pass",
        summary="summary",
        score=score,
        source=source,
    )


if __name__ == "__main__":
    unittest.main()
