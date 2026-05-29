import json
import re
from typing import Any

from neo4j import GraphDatabase

from .schema import GraphEdge, GraphNode

SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Neo4jGraphStore:
    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        reset: bool = False,
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._node_buffer: dict[str, GraphNode] = {}
        self._edge_buffer: dict[str, GraphEdge] = {}
        self.driver.verify_connectivity()
        self._ensure_constraints()
        if reset:
            self.clear()

    def clear(self) -> None:
        self._execute_write("MATCH (n:GraphNode) DETACH DELETE n")

    def upsert_node(self, node: GraphNode) -> None:
        self._node_buffer[node.node_id] = node

    def upsert_edge(self, edge: GraphEdge) -> None:
        safe_graph_name(edge.edge_type)
        self._edge_buffer[edge.edge_id] = edge

    def commit(self) -> None:
        self._flush_nodes()
        self._flush_edges()

    def close(self) -> None:
        self.driver.close()

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        rows = self._execute_read(
            """
            MATCH (n:GraphNode {node_id: $node_id})
            RETURN n.node_id AS node_id,
                   n.node_type AS node_type,
                   n.name AS name,
                   n.metadata_json AS metadata_json
            """,
            node_id=node_id,
        )
        return row_to_node(rows[0]) if rows else None

    def count_nodes(self) -> int:
        rows = self._execute_read("MATCH (n:GraphNode) RETURN count(n) AS count")
        return int(rows[0]["count"])

    def count_edges(self) -> int:
        rows = self._execute_read("MATCH (:GraphNode)-[r]->(:GraphNode) RETURN count(r) AS count")
        return int(rows[0]["count"])

    def edges_for_node(self, node_id: str, edge_types: list[str] | None = None) -> list[dict[str, Any]]:
        rows = self._execute_read(
            """
            MATCH (a:GraphNode {node_id: $node_id})-[r]-(b:GraphNode)
            WHERE $edge_types IS NULL OR type(r) IN $edge_types
            RETURN r.edge_id AS edge_id,
                   startNode(r).node_id AS source_id,
                   endNode(r).node_id AS target_id,
                   type(r) AS edge_type,
                   r.metadata_json AS metadata_json
            ORDER BY edge_type, edge_id
            """,
            node_id=node_id,
            edge_types=edge_types,
        )
        return [row_to_edge(row) for row in rows]

    def nodes_by_ids(self, node_ids: list[str]) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        rows = self._execute_read(
            """
            MATCH (n:GraphNode)
            WHERE n.node_id IN $node_ids
            RETURN n.node_id AS node_id,
                   n.node_type AS node_type,
                   n.name AS name,
                   n.metadata_json AS metadata_json
            ORDER BY node_type, name
            """,
            node_ids=node_ids,
        )
        return [row_to_node(row) for row in rows]

    def _ensure_constraints(self) -> None:
        self._execute_write(
            """
            CREATE CONSTRAINT graph_node_id IF NOT EXISTS
            FOR (n:GraphNode)
            REQUIRE n.node_id IS UNIQUE
            """
        )

    def _flush_nodes(self) -> None:
        records = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "name": node.name,
                "metadata_json": json.dumps(node.metadata, ensure_ascii=False),
            }
            for node in self._node_buffer.values()
        ]
        for batch in chunks(records, 500):
            self._execute_write(
                """
                UNWIND $rows AS row
                MERGE (n:GraphNode {node_id: row.node_id})
                SET n.node_type = row.node_type,
                    n.name = row.name,
                    n.metadata_json = row.metadata_json
                """,
                rows=batch,
            )
        self._node_buffer.clear()

    def _flush_edges(self) -> None:
        grouped: dict[str, list[GraphEdge]] = {}
        for edge in self._edge_buffer.values():
            grouped.setdefault(safe_graph_name(edge.edge_type), []).append(edge)

        for edge_type, edges in grouped.items():
            records = [
                {
                    "edge_id": edge.edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type,
                    "metadata_json": json.dumps(edge.metadata, ensure_ascii=False),
                }
                for edge in edges
            ]
            for batch in chunks(records, 500):
                self._execute_write(
                    f"""
                    UNWIND $rows AS row
                    MATCH (source:GraphNode {{node_id: row.source_id}})
                    MATCH (target:GraphNode {{node_id: row.target_id}})
                    MERGE (source)-[r:{edge_type} {{edge_id: row.edge_id}}]->(target)
                    SET r.source_id = row.source_id,
                        r.target_id = row.target_id,
                        r.edge_type = row.edge_type,
                        r.metadata_json = row.metadata_json
                    """,
                    rows=batch,
                )
        self._edge_buffer.clear()

    def _execute_write(self, query: str, **params: Any) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, **params).consume())

    def _execute_read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(lambda tx: [dict(row) for row in tx.run(query, **params)])


def safe_graph_name(name: str) -> str:
    if not SAFE_NAME_RE.match(name):
        raise ValueError(f"Unsafe Neo4j label/type name: {name}")
    return name


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def row_to_node(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": row["node_id"],
        "node_type": row["node_type"],
        "name": row["name"],
        "metadata": json.loads(row["metadata_json"]),
    }


def row_to_edge(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": row["edge_id"],
        "source_id": row["source_id"],
        "target_id": row["target_id"],
        "edge_type": row["edge_type"],
        "metadata": json.loads(row["metadata_json"]),
    }
