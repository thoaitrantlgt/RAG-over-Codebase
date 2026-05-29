from typing import Any

from .chunk import create_chunk
from .schema import CodeChunk
from .source import child_by_field_or_type, named_children, node_text

FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
}

METHOD_TYPES = {
    "method_definition",
    "public_field_definition",
}


def extract_typescript_chunks(
    *,
    repo: str,
    relative_path: str,
    language: str,
    source_bytes: bytes,
    root_node: Any,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    def visit(node: Any, class_stack: list[str]) -> None:
        actual_node = unwrap_export(node)

        if actual_node.type == "class_declaration":
            name = get_node_name(source_bytes, actual_node)
            if name:
                qualified_name = ".".join([*class_stack, name])
                chunks.append(
                    create_chunk(
                        repo=repo,
                        relative_path=relative_path,
                        language=language,
                        symbol_name=qualified_name,
                        symbol_kind="class",
                        source_bytes=source_bytes,
                        node=actual_node,
                    )
                )

                for child in named_children(actual_node):
                    visit(child, [*class_stack, name])
                return

        if actual_node.type in FUNCTION_TYPES:
            name = get_node_name(source_bytes, actual_node)
            if name:
                chunks.append(
                    create_chunk(
                        repo=repo,
                        relative_path=relative_path,
                        language=language,
                        symbol_name=name,
                        symbol_kind="function",
                        source_bytes=source_bytes,
                        node=actual_node,
                    )
                )
            return

        if actual_node.type in METHOD_TYPES and class_stack:
            name = get_node_name(source_bytes, actual_node)
            if name:
                chunks.append(
                    create_chunk(
                        repo=repo,
                        relative_path=relative_path,
                        language=language,
                        symbol_name=".".join([*class_stack, name]),
                        symbol_kind="method",
                        source_bytes=source_bytes,
                        node=actual_node,
                    )
                )
            return

        for child in named_children(actual_node):
            visit(child, class_stack)

    visit(root_node, [])
    return chunks


def unwrap_export(node: Any) -> Any:
    if node.type != "export_statement":
        return node

    for child in named_children(node):
        if child.type != "export_clause":
            return child

    return node


def get_node_name(source_bytes: bytes, node: Any) -> str:
    name_node = child_by_field_or_type(
        node,
        "name",
        ("identifier", "type_identifier", "property_identifier"),
    )
    return node_text(source_bytes, name_node) if name_node is not None else ""
