from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    name: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
