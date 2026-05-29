import argparse
import json

from dataclasses import replace

from packages.agent.config import AgentConfig, config_from_env

from .repobench_r import load_records, prepare_repobench_records, run_repobench_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RepoBench-R retrieval benchmark utilities.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser(
        "prepare",
        help="Convert RepoBench-R records into Phase 1-like chunks and eval queries.",
    )
    add_dataset_args(prepare)
    prepare.add_argument("--chunks-out", default="data/evals/repobench_r_chunks.jsonl")
    prepare.add_argument("--queries-out", default="data/evals/repobench_r_queries.jsonl")
    prepare.add_argument("--repo-name", default="repobench-r")
    prepare.add_argument("--max-query-chars", type=int, default=4000)

    run = subcommands.add_parser("run", help="Run retrieval eval over prepared queries.")
    run.add_argument("--queries", default="data/evals/repobench_r_queries.jsonl")
    run.add_argument("--report-out", default="data/evals/repobench_r_report.json")
    run.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    run.add_argument("--top-k", type=int, default=10)
    add_retrieval_args(run)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local RepoBench-R JSON or JSONL file.")
    source.add_argument("--hf-config", help="HuggingFace config, e.g. python_cff or python_cfr.")
    parser.add_argument("--split", default="test_easy")
    parser.add_argument("--limit", type=int)


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dense-backend", choices=["local", "qdrant"], default=None)
    parser.add_argument("--dense-index")
    parser.add_argument("--sparse-backend", choices=["sqlite", "tantivy"], default=None)
    parser.add_argument("--sparse-index")
    parser.add_argument("--qdrant-collection")
    parser.add_argument("--embedding-provider", choices=["hashing", "fastembed"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-cache-dir")
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--retrieve-top-k", type=int)
    parser.add_argument("--fused-top-k", type=int)
    parser.add_argument("--rrf-k", type=int)


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "prepare":
        records = load_records(
            input_path=args.input,
            hf_config=args.hf_config,
            split=args.split,
            limit=args.limit,
        )
        result = prepare_repobench_records(
            records,
            chunks_path=args.chunks_out,
            queries_path=args.queries_out,
            repo_name=args.repo_name,
            max_query_chars=args.max_query_chars,
        )
        print(json.dumps({
            "sample_count": result.sample_count,
            "chunk_count": result.chunk_count,
            "query_count": result.query_count,
            "chunks_path": str(result.chunks_path),
            "queries_path": str(result.queries_path),
        }, ensure_ascii=False, indent=2))
        return

    config = apply_retrieval_overrides(config_from_env(), args, top_k=args.top_k)
    result = run_repobench_eval(
        queries_path=args.queries,
        config=config,
        mode=args.mode,
        top_k=args.top_k,
    )
    write_json(args.report_out, result.to_dict())
    print(json.dumps(result.to_dict()["metrics"], ensure_ascii=False, indent=2))
    print(f"Report: {args.report_out}")


def apply_retrieval_overrides(config: AgentConfig, args, *, top_k: int) -> AgentConfig:
    updates = {
        "retrieve_top_k": args.retrieve_top_k or max(top_k, config.retrieve_top_k),
        "fused_top_k": args.fused_top_k or max(top_k, config.fused_top_k),
    }
    for arg_name, field_name in [
        ("dense_backend", "dense_backend"),
        ("dense_index", "dense_index"),
        ("sparse_backend", "sparse_backend"),
        ("sparse_index", "sparse_index"),
        ("qdrant_collection", "qdrant_collection"),
        ("embedding_provider", "embedding_provider"),
        ("embedding_model", "embedding_model"),
        ("embedding_cache_dir", "embedding_cache_dir"),
        ("embedding_dimensions", "embedding_dimensions"),
        ("rrf_k", "rrf_k"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    return replace(config, **updates)


def write_json(path: str, payload: dict) -> None:
    from pathlib import Path

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
