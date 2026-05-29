from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packages.ingestion.parser import parse_source
from packages.ingestion.source import child_by_field_or_type, line_range, named_children, node_text


@dataclass(frozen=True)
class ImportRef:
    name: str
    line: int
    target_path: str | None
    raw: str


@dataclass(frozen=True)
class CallRef:
    name: str
    line: int
    raw: str


@dataclass(frozen=True)
class FileRelations:
    imports: list[ImportRef]
    calls: list[CallRef]


def analyze_file(repo_root: Path, file_path: Path, language: str) -> FileRelations:
    source_bytes = file_path.read_bytes()
    parsed = parse_source(source_bytes, language)
    imports: list[ImportRef] = []
    calls: list[CallRef] = []

    def visit(node: Any) -> None:
        if language == "python":
            import_ref = python_import_ref(repo_root, file_path, source_bytes, node)
            if import_ref is not None:
                imports.append(import_ref)
        else:
            import_ref = typescript_import_ref(repo_root, file_path, source_bytes, node)
            if import_ref is not None:
                imports.append(import_ref)

        call_ref = call_ref_for_node(source_bytes, node, language)
        if call_ref is not None:
            calls.append(call_ref)

        for child in named_children(node):
            visit(child)

    visit(parsed.tree.root_node)
    return FileRelations(imports=imports, calls=calls)


def python_import_ref(repo_root: Path, file_path: Path, source_bytes: bytes, node: Any) -> ImportRef | None:
    if node.type not in {"import_statement", "import_from_statement"}:
        return None

    start_line, _ = line_range(node)
    raw = node_text(source_bytes, node)
    name = raw

    if node.type == "import_statement":
        name_node = first_named_descendant(node, {"dotted_name", "identifier"})
        if name_node is not None:
            name = node_text(source_bytes, name_node)
    else:
        module_node = first_named_descendant(node, {"dotted_name", "relative_import", "identifier"})
        if module_node is not None:
            name = node_text(source_bytes, module_node)

    return ImportRef(
        name=name.lstrip(".") or name,
        line=start_line,
        target_path=resolve_python_module(repo_root, file_path, name),
        raw=raw,
    )


def typescript_import_ref(repo_root: Path, file_path: Path, source_bytes: bytes, node: Any) -> ImportRef | None:
    if node.type != "import_statement":
        return None

    start_line, _ = line_range(node)
    raw = node_text(source_bytes, node)
    source_node = first_named_descendant(node, {"string"})
    if source_node is None:
        return ImportRef(name=raw, line=start_line, target_path=None, raw=raw)

    name = node_text(source_bytes, source_node).strip("\"'")
    return ImportRef(
        name=name,
        line=start_line,
        target_path=resolve_typescript_module(repo_root, file_path, name),
        raw=raw,
    )


def call_ref_for_node(source_bytes: bytes, node: Any, language: str) -> CallRef | None:
    if language == "python" and node.type != "call":
        return None
    if language != "python" and node.type != "call_expression":
        return None

    start_line, _ = line_range(node)
    callee = node.child_by_field_name("function")
    if callee is None:
        callee = named_children(node)[0] if named_children(node) else None
    if callee is None:
        return None

    name = callable_name(source_bytes, callee)
    if not name:
        return None

    return CallRef(
        name=name,
        line=start_line,
        raw=node_text(source_bytes, node),
    )


def callable_name(source_bytes: bytes, node: Any) -> str:
    if node.type in {"identifier", "property_identifier"}:
        return node_text(source_bytes, node)

    if node.type in {"attribute", "member_expression"}:
        parts = [
            node_text(source_bytes, child)
            for child in named_children(node)
            if child.type in {"identifier", "property_identifier"}
        ]
        return ".".join(parts) if parts else node_text(source_bytes, node)

    name_node = child_by_field_or_type(
        node,
        "name",
        ("identifier", "property_identifier"),
    )
    if name_node is not None:
        return node_text(source_bytes, name_node)

    return ""


def resolve_python_module(repo_root: Path, file_path: Path, module_name: str) -> str | None:
    if module_name.startswith("."):
        dots = len(module_name) - len(module_name.lstrip("."))
        base = file_path.parent
        for _ in range(max(dots - 1, 0)):
            base = base.parent
        module_name = module_name.lstrip(".")
        module_path = module_name.replace(".", "/")
        candidates = [
            base / f"{module_path}.py",
            base / module_path / "__init__.py",
        ]
    else:
        module_path = module_name.replace(".", "/")
        candidates = [
            repo_root / f"{module_path}.py",
            repo_root / module_path / "__init__.py",
        ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_relative_to(repo_root):
            return candidate.relative_to(repo_root).as_posix()
    return None


def resolve_typescript_module(repo_root: Path, file_path: Path, module_name: str) -> str | None:
    if not module_name.startswith("."):
        return None

    base = (file_path.parent / module_name).resolve()
    candidates = [
        base.with_suffix(".ts"),
        base.with_suffix(".tsx"),
        base / "index.ts",
        base / "index.tsx",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_relative_to(repo_root):
            return candidate.relative_to(repo_root).as_posix()
    return None


def first_named_descendant(node: Any, types: set[str]) -> Any | None:
    for child in named_children(node):
        if child.type in types:
            return child
        found = first_named_descendant(child, types)
        if found is not None:
            return found
    return None
