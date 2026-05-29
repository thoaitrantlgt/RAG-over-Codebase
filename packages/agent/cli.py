import argparse
import json
import sys
from dataclasses import replace

from .config import config_from_env
from .workflow import run_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4 agentic code search.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ask = subcommands.add_parser("ask", help="Ask a question over the indexed codebase.")
    ask.add_argument("--q", required=True, help="Question to answer.")
    ask.add_argument("--answer-only", action="store_true", help="Print only the synthesized answer.")
    ask.add_argument("--include-debug", action="store_true")
    ask.add_argument("--no-synth", action="store_true", help="Skip LM Studio and return context only.")
    ask.add_argument("--dense-index")
    ask.add_argument("--sparse-index")
    ask.add_argument("--graph-path")
    ask.add_argument("--embedding-provider")
    ask.add_argument("--embedding-model")
    ask.add_argument("--embedding-cache-dir")
    ask.add_argument("--qdrant-collection")
    ask.add_argument("--lmstudio-base-url")
    ask.add_argument("--lmstudio-model")
    ask.add_argument("--retrieve-top-k", type=int)
    ask.add_argument("--fused-top-k", type=int)
    ask.add_argument("--context-top-k", type=int)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(config_from_env(), args)
    result = run_agent(args.q, config, synthesize=not args.no_synth)
    if args.answer_only:
        sys.stdout.buffer.write(result.answer.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        return
    write_json(result.to_dict(include_debug=args.include_debug))


def apply_overrides(config, args):
    updates = {}
    for arg_name, field_name in [
        ("dense_index", "dense_index"),
        ("sparse_index", "sparse_index"),
        ("graph_path", "graph_path"),
        ("embedding_provider", "embedding_provider"),
        ("embedding_model", "embedding_model"),
        ("embedding_cache_dir", "embedding_cache_dir"),
        ("qdrant_collection", "qdrant_collection"),
        ("lmstudio_base_url", "lmstudio_base_url"),
        ("lmstudio_model", "lmstudio_model"),
        ("retrieve_top_k", "retrieve_top_k"),
        ("fused_top_k", "fused_top_k"),
        ("context_top_k", "context_top_k"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    return replace(config, **updates)


def write_json(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
