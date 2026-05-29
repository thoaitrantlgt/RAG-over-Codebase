import tempfile
import unittest
from pathlib import Path

from packages.graph import build_graph, expand_context
from packages.graph.ids import file_node_id, symbol_node_id
from packages.graph.store import SQLiteGraphStore
from packages.ingestion import ingest_repository, write_jsonl


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


class GraphTests(unittest.TestCase):
    def test_build_graph_creates_nodes_and_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths, chunks = self._build_graph(Path(temp_dir))
            store = SQLiteGraphStore(paths["graph"])
            try:
                self.assertGreaterEqual(store.count_nodes(), 14)
                self.assertGreaterEqual(store.count_edges(), 14)

                auth_class = next(chunk for chunk in chunks if chunk.symbol_name == "AuthService")
                auth_method = next(
                    chunk for chunk in chunks if chunk.symbol_name == "AuthService.verify_token"
                )
                edges = store.edges_for_node(symbol_node_id(auth_method.chunk_id), ["CONTAINS"])
            finally:
                store.close()

        class_to_method = [
            edge
            for edge in edges
            if edge["source_id"] == symbol_node_id(auth_class.chunk_id)
            and edge["target_id"] == symbol_node_id(auth_method.chunk_id)
        ]
        self.assertEqual(len(class_to_method), 1)

    def test_graph_resolves_local_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths, chunks = self._build_graph(Path(temp_dir))
            caller = next(chunk for chunk in chunks if chunk.symbol_name == "UserService.findUser")
            target = next(chunk for chunk in chunks if chunk.symbol_name == "getUser")

            store = SQLiteGraphStore(paths["graph"])
            try:
                edges = store.edges_for_node(symbol_node_id(caller.chunk_id), ["CALLS"])
            finally:
                store.close()

        matches = [
            edge
            for edge in edges
            if edge["source_id"] == symbol_node_id(caller.chunk_id)
            and edge["target_id"] == symbol_node_id(target.chunk_id)
            and edge["metadata"]["confidence"] == "resolved"
        ]
        self.assertEqual(len(matches), 1)

    def test_graph_keeps_unresolved_imports_as_external_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths, _ = self._build_graph(Path(temp_dir))
            store = SQLiteGraphStore(paths["graph"])
            try:
                edges = store.edges_for_node(
                    file_node_id("sample_repo", "src/auth/service.py"),
                    ["IMPORTS"],
                )
                external_nodes = [
                    store.get_node(edge["target_id"])
                    for edge in edges
                    if edge["metadata"]["confidence"] == "unresolved"
                ]
            finally:
                store.close()

        names = [node["name"] for node in external_nodes if node]
        self.assertIn("jwt", names)

    def test_expand_context_returns_related_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths, chunks = self._build_graph(Path(temp_dir))
            method = next(chunk for chunk in chunks if chunk.symbol_name == "AuthService.verify_token")
            expanded = expand_context(
                graph_path=paths["graph"],
                chunk_id=method.chunk_id,
                max_nodes=10,
            )

        names = [node["name"] for node in expanded["related"]]
        self.assertIn("AuthService", names)
        self.assertIn("src/auth/service.py", names)

    def _build_graph(self, root: Path) -> tuple[dict[str, Path], list]:
        chunks_path = root / "chunks.jsonl"
        graph_path = root / "code_graph.sqlite"
        result = ingest_repository(repo_path=FIXTURE_REPO, repo_name="sample_repo")
        write_jsonl(chunks_path, result.chunks)
        graph_result = build_graph(
            repo_path=FIXTURE_REPO,
            repo_name="sample_repo",
            chunks_path=chunks_path,
            graph_path=graph_path,
        )
        self.assertGreater(graph_result.node_count, 0)
        return {"chunks": chunks_path, "graph": graph_path}, result.chunks


if __name__ == "__main__":
    unittest.main()
