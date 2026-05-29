from typing import Any

from .chunk import create_chunk
from .schema import CodeChunk
from .source import child_by_field_or_type, named_children, node_text


def extract_python_chunks(
    *,
    repo: str,
    relative_path: str,
    language: str,
    source_bytes: bytes,
    root_node: Any,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    def visit(node: Any, class_stack: list[str]) -> None:
        if node.type == "class_definition":
            name = get_python_name(source_bytes, node)
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
                        node=node,
                    )
                )

                for child in named_children(node):
                    visit(child, [*class_stack, name])
                return

        if node.type == "function_definition":
            name = get_python_name(source_bytes, node)
            if name:
                chunks.append(
                    create_chunk(
                        repo=repo,
                        relative_path=relative_path,
                        language=language,
                        symbol_name=".".join([*class_stack, name]),
                        symbol_kind="method" if class_stack else "function",
                        source_bytes=source_bytes,
                        node=node,
                    )
                )
            return

        for child in named_children(node):
            visit(child, class_stack)

    visit(root_node, [])
    return chunks


def get_python_name(source_bytes: bytes, node: Any) -> str:
    name_node = child_by_field_or_type(node, "name", ("identifier",))
    return node_text(source_bytes, name_node) if name_node is not None else ""
