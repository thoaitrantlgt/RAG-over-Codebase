from pathlib import Path
from typing import Any

from .ids import file_node_id, symbol_node_id
from .builder import create_graph_store
from .store import SQLiteGraphStore


def expand_context(
    *,
    graph_path: str | Path,
    chunk_id: str,
    max_nodes: int = 20,
    backend: str = "sqlite",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
) -> dict[str, Any]:
    store = create_graph_store(
        backend=backend,
        graph_path=graph_path,
        reset=False,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
    )
    try:
        seed_id = symbol_node_id(chunk_id)
        seed = store.get_node(seed_id)
        if seed is None:
            return {"seed": None, "related": [], "edges": []}

        related_ids: list[str] = []
        edges = store.edges_for_node(seed_id, ["CALLS", "CONTAINS"])

        seed_meta = seed["metadata"]
        file_id = file_node_id(seed_meta["repo"], seed_meta["path"])
        edges.extend(store.edges_for_node(file_id, ["IMPORTS", "CONTAINS"]))
        edges = dedupe_edges(edges)

        for edge in edges:
            for node_id in (edge["source_id"], edge["target_id"]):
                if node_id != seed_id and node_id not in related_ids:
                    related_ids.append(node_id)

        related = store.nodes_by_ids(related_ids[:max_nodes])
        return {"seed": seed, "related": related, "edges": edges}
    finally:
        store.close()


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for edge in edges:
        if edge["edge_id"] in seen:
            continue
        seen.add(edge["edge_id"])
        result.append(edge)
    return result
