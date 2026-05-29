from dataclasses import dataclass
from typing import Any

from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_typescript


@dataclass(frozen=True)
class ParsedSource:
    tree: Any
    has_error: bool


def parse_source(source_bytes: bytes, language: str) -> ParsedSource:
    parser = Parser()
    parser.language = language_for(language)
    tree = parser.parse(source_bytes)
    return ParsedSource(tree=tree, has_error=bool(tree.root_node.has_error))


def language_for(language: str) -> Language:
    if language == "python":
        return Language(tree_sitter_python.language())
    if language == "typescript":
        return Language(tree_sitter_typescript.language_typescript())
    if language == "tsx":
        return Language(tree_sitter_typescript.language_tsx())
    raise ValueError(f"Unsupported language: {language}")
