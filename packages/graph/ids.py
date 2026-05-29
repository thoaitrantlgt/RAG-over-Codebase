import hashlib


def repo_node_id(repo: str) -> str:
    return f"repo:{repo}"


def file_node_id(repo: str, path: str) -> str:
    return f"file:{repo}:{path}"


def symbol_node_id(chunk_id: str) -> str:
    return f"symbol:{chunk_id}"


def external_node_id(repo: str, name: str) -> str:
    return f"external:{repo}:{name}"


def edge_id(source_id: str, target_id: str, edge_type: str, line: int | None = None) -> str:
    raw = f"{source_id}|{target_id}|{edge_type}|{line or ''}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"
