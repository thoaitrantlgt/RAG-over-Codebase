from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packages.indexing.io import read_jsonl
from packages.ingestion.walker import walk_repository

from .analysis import analyze_file
from .ids import edge_id, external_node_id, file_node_id, repo_node_id, symbol_node_id
from .resolver import SymbolResolver
from .schema import GraphEdge, GraphNode
from .neo4j_store import Neo4jGraphStore
from .store import SQLiteGraphStore


@dataclass(frozen=True)
class GraphBuildResult:
    graph_path: Path
    node_count: int
    edge_count: int


def build_graph(
    *,
    repo_path: str | Path,
    repo_name: str,
    chunks_path: str | Path,
    graph_path: str | Path,
    backend: str = "sqlite",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
) -> GraphBuildResult:
    root_path = Path(repo_path).resolve()
    chunks = read_jsonl(chunks_path)
    resolver = SymbolResolver(chunks)
    store = create_graph_store(
        backend=backend,
        graph_path=graph_path,
        reset=True,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
    )

    try:
        add_static_nodes_and_contains_edges(store, repo_name, chunks)
        add_ast_edges(store, root_path, repo_name, resolver)
        store.commit()
        return GraphBuildResult(
            graph_path=Path(graph_path).resolve(),
            node_count=store.count_nodes(),
            edge_count=store.count_edges(),
        )
    finally:
        store.close()


def create_graph_store(
    *,
    backend: str,
    graph_path: str | Path,
    reset: bool,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
):
    if backend == "sqlite":
        return SQLiteGraphStore(graph_path, reset=reset)
    if backend == "neo4j":
        if not neo4j_uri or not neo4j_user or not neo4j_password:
            raise ValueError("Neo4j backend requires URI, user, and password")
        return Neo4jGraphStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
            reset=reset,
        )
    raise ValueError(f"Unsupported graph backend: {backend}")


def add_static_nodes_and_contains_edges(
    store: SQLiteGraphStore,
    repo_name: str,
    chunks: list[dict[str, Any]],
) -> None:
    repo_id = repo_node_id(repo_name)
    store.upsert_node(
        GraphNode(
            node_id=repo_id,
            node_type="Repo",
            name=repo_name,
            metadata={"repo": repo_name},
        )
    )

    paths = sorted({chunk["path"] for chunk in chunks})
    for path in paths:
        file_id = file_node_id(repo_name, path)
        store.upsert_node(
            GraphNode(
                node_id=file_id,
                node_type="File",
                name=path,
                metadata={"repo": repo_name, "path": path},
            )
        )
        store.upsert_edge(
            GraphEdge(
                edge_id=edge_id(repo_id, file_id, "CONTAINS"),
                source_id=repo_id,
                target_id=file_id,
                edge_type="CONTAINS",
                metadata={"confidence": "resolved"},
            )
        )

    for chunk in chunks:
        node_type = "Class" if chunk["symbol_kind"] == "class" else "Function"
        current_symbol_id = symbol_node_id(chunk["chunk_id"])
        current_file_id = file_node_id(repo_name, chunk["path"])
        store.upsert_node(
            GraphNode(
                node_id=current_symbol_id,
                node_type=node_type,
                name=chunk["symbol_name"],
                metadata=chunk,
            )
        )
        store.upsert_edge(
            GraphEdge(
                edge_id=edge_id(current_file_id, current_symbol_id, "CONTAINS"),
                source_id=current_file_id,
                target_id=current_symbol_id,
                edge_type="CONTAINS",
                metadata={"confidence": "resolved"},
            )
        )

    classes = [chunk for chunk in chunks if chunk["symbol_kind"] == "class"]
    methods = [chunk for chunk in chunks if chunk["symbol_kind"] == "method"]
    for class_chunk in classes:
        for method_chunk in methods:
            if method_chunk["path"] != class_chunk["path"]:
                continue
            if not method_chunk["symbol_name"].startswith(f"{class_chunk['symbol_name']}."):
                continue
            class_id = symbol_node_id(class_chunk["chunk_id"])
            method_id = symbol_node_id(method_chunk["chunk_id"])
            store.upsert_edge(
                GraphEdge(
                    edge_id=edge_id(class_id, method_id, "CONTAINS"),
                    source_id=class_id,
                    target_id=method_id,
                    edge_type="CONTAINS",
                    metadata={"confidence": "resolved"},
                )
            )


def add_ast_edges(
    store: SQLiteGraphStore,
    root_path: Path,
    repo_name: str,
    resolver: SymbolResolver,
) -> None:
    for source_file in walk_repository(root_path):
        relative_path = source_file.path.relative_to(root_path).as_posix()
        relations = analyze_file(root_path, source_file.path, source_file.language)
        source_file_id = file_node_id(repo_name, relative_path)

        for import_ref in relations.imports:
            if import_ref.target_path is not None:
                target_id = file_node_id(repo_name, import_ref.target_path)
                confidence = "resolved"
            else:
                target_id = external_node_id(repo_name, import_ref.name)
                confidence = "unresolved"
                store.upsert_node(
                    GraphNode(
                        node_id=target_id,
                        node_type="External",
                        name=import_ref.name,
                        metadata={"repo": repo_name, "name": import_ref.name},
                    )
                )

            store.upsert_edge(
                GraphEdge(
                    edge_id=edge_id(source_file_id, target_id, "IMPORTS", import_ref.line),
                    source_id=source_file_id,
                    target_id=target_id,
                    edge_type="IMPORTS",
                    metadata={
                        "line": import_ref.line,
                        "raw": import_ref.raw,
                        "confidence": confidence,
                    },
                )
            )

        for call_ref in relations.calls:
            source_symbol = resolver.enclosing_callable(relative_path, call_ref.line)
            if source_symbol is None:
                continue

            target_symbol = resolver.resolve_call(relative_path, call_ref.name)
            source_id = symbol_node_id(source_symbol.chunk_id)
            if target_symbol is not None:
                target_id = symbol_node_id(target_symbol.chunk_id)
                confidence = "resolved"
            else:
                target_id = external_node_id(repo_name, call_ref.name)
                confidence = "unresolved"
                store.upsert_node(
                    GraphNode(
                        node_id=target_id,
                        node_type="External",
                        name=call_ref.name,
                        metadata={"repo": repo_name, "name": call_ref.name},
                    )
                )

            if source_id == target_id:
                continue

            store.upsert_edge(
                GraphEdge(
                    edge_id=edge_id(source_id, target_id, "CALLS", call_ref.line),
                    source_id=source_id,
                    target_id=target_id,
                    edge_type="CALLS",
                    metadata={
                        "line": call_ref.line,
                        "raw": call_ref.raw,
                        "confidence": confidence,
                    },
                )
            )
