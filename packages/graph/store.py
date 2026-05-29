import json
import sqlite3
from pathlib import Path
from typing import Any

from .schema import GraphEdge, GraphNode


class SQLiteGraphStore:
    def __init__(self, path: str | Path, reset: bool = False) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()
        if reset:
            self.clear()

    def clear(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM edges")
            self.connection.execute("DELETE FROM nodes")

    def upsert_node(self, node: GraphNode) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO nodes (node_id, node_type, name, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                node.node_id,
                node.node_type,
                node.name,
                json.dumps(node.metadata, ensure_ascii=False),
            ),
        )

    def upsert_edge(self, edge: GraphEdge) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO edges (
                edge_id,
                source_id,
                target_id,
                edge_type,
                metadata
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                json.dumps(edge.metadata, ensure_ascii=False),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT node_id, node_type, name, metadata FROM nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return row_to_node(row) if row else None

    def count_nodes(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    def count_edges(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    def edges_for_node(self, node_id: str, edge_types: list[str] | None = None) -> list[dict[str, Any]]:
        params: list[Any] = [node_id, node_id]
        where = "(source_id = ? OR target_id = ?)"
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            where += f" AND edge_type IN ({placeholders})"
            params.extend(edge_types)
        rows = self.connection.execute(
            f"""
            SELECT edge_id, source_id, target_id, edge_type, metadata
            FROM edges
            WHERE {where}
            ORDER BY edge_type, edge_id
            """,
            params,
        ).fetchall()
        return [row_to_edge(row) for row in rows]

    def nodes_by_ids(self, node_ids: list[str]) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.connection.execute(
            f"""
            SELECT node_id, node_type, name, metadata
            FROM nodes
            WHERE node_id IN ({placeholders})
            ORDER BY node_type, name
            """,
            node_ids,
        ).fetchall()
        return [row_to_node(row) for row in rows]

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
        )
        self.connection.commit()


def row_to_node(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "node_id": row["node_id"],
        "node_type": row["node_type"],
        "name": row["name"],
        "metadata": json.loads(row["metadata"]),
    }


def row_to_edge(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "edge_id": row["edge_id"],
        "source_id": row["source_id"],
        "target_id": row["target_id"],
        "edge_type": row["edge_type"],
        "metadata": json.loads(row["metadata"]),
    }
