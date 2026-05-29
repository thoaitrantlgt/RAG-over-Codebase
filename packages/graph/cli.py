import argparse
import json
import os
from pathlib import Path

from .builder import build_graph
from .expansion import expand_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query the Phase 3 symbol graph.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="Build graph from a repo and Phase 1 chunks.")
    build.add_argument("--repo-path", required=True)
    build.add_argument("--repo-name", required=True)
    build.add_argument("--chunks", required=True)
    build.add_argument("--graph", default="data/graph/code_graph.sqlite")
    add_backend_args(build)

    expand = subcommands.add_parser("expand", help="Expand graph context around a chunk.")
    expand.add_argument("--graph", default="data/graph/code_graph.sqlite")
    expand.add_argument("--chunk-id", required=True)
    expand.add_argument("--max-nodes", type=int, default=20)
    add_backend_args(expand)

    return parser


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["sqlite", "neo4j"], default="sqlite")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE"))


def main() -> None:
    load_env()
    args = build_parser().parse_args()

    if args.command == "build":
        result = build_graph(
            repo_path=args.repo_path,
            repo_name=args.repo_name,
            chunks_path=args.chunks,
            graph_path=args.graph,
            backend=args.backend,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
        )
        if args.backend == "neo4j":
            database = args.neo4j_database or "default"
            print(f"Neo4j: {args.neo4j_uri} database={database}")
        else:
            print(f"Graph: {result.graph_path}")
        print(f"Nodes: {result.node_count}")
        print(f"Edges: {result.edge_count}")
        return

    result = expand_context(
        graph_path=args.graph,
        chunk_id=args.chunk_id,
        max_nodes=args.max_nodes,
        backend=args.backend,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


if __name__ == "__main__":
    main()
