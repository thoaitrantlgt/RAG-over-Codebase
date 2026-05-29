from typing import Any


def point_row(point: Any) -> int:
    if hasattr(point, "row"):
        return int(point.row)
    return int(point[0])


def line_range(node: Any) -> tuple[int, int]:
    return point_row(node.start_point) + 1, point_row(node.end_point) + 1


def node_text(source_bytes: bytes, node: Any) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def named_children(node: Any) -> list[Any]:
    if hasattr(node, "named_children"):
        return list(node.named_children)
    return [child for child in node.children if child.is_named]


def child_by_type(node: Any, node_type: str) -> Any | None:
    for child in named_children(node):
        if child.type == node_type:
            return child
    return None


def child_by_field_or_type(node: Any, field_name: str, types: tuple[str, ...]) -> Any | None:
    child = node.child_by_field_name(field_name)
    if child is not None:
        return child

    for node_type in types:
        child = child_by_type(node, node_type)
        if child is not None:
            return child

    return None
