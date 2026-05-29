import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    dense_backend: str = "qdrant"
    dense_index: str = "data/index/qdrant_rag_project_fastembed"
    sparse_backend: str = "tantivy"
    sparse_index: str = "data/index/tantivy_rag_project_fastembed"
    qdrant_collection: str = "code_chunks"
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = "data/models/fastembed"
    embedding_dimensions: int = 128
    graph_backend: str = "sqlite"
    graph_path: str = "data/graph/rag_project.sqlite"
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str | None = None
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "qwen3-1.7b"
    lmstudio_timeout_seconds: int = 120
    retrieve_top_k: int = 25
    fused_top_k: int = 10
    context_top_k: int = 10
    graph_max_nodes: int = 12
    rrf_k: int = 60


def load_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def config_from_env() -> AgentConfig:
    load_env()
    return AgentConfig(
        dense_backend=os.getenv("DENSE_BACKEND", "qdrant"),
        dense_index=os.getenv("QDRANT_PATH", "data/index/qdrant_rag_project_fastembed"),
        sparse_backend=os.getenv("SPARSE_BACKEND", "tantivy"),
        sparse_index=os.getenv("TANTIVY_INDEX_PATH", "data/index/tantivy_rag_project_fastembed"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "code_chunks"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "fastembed"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_cache_dir=os.getenv("EMBEDDING_CACHE_DIR", "data/models/fastembed"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "128")),
        graph_backend=os.getenv("GRAPH_BACKEND", "sqlite"),
        graph_path=os.getenv("GRAPH_PATH", "data/graph/rag_project.sqlite"),
        neo4j_uri=os.getenv("NEO4J_URI") or None,
        neo4j_user=os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD") or None,
        neo4j_database=os.getenv("NEO4J_DATABASE") or None,
        lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        lmstudio_model=os.getenv("LMSTUDIO_MODEL", "qwen3-1.7b"),
        lmstudio_timeout_seconds=int(os.getenv("LMSTUDIO_TIMEOUT_SECONDS", "120")),
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", os.getenv("TOP_K", "25"))),
        fused_top_k=int(os.getenv("FUSED_TOP_K", "10")),
        context_top_k=int(os.getenv("CONTEXT_TOP_K", "10")),
        graph_max_nodes=int(os.getenv("GRAPH_MAX_NODES", "12")),
        rrf_k=int(os.getenv("RRF_K", "60")),
    )
