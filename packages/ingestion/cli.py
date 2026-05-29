import argparse
from pathlib import Path

from .ingest import ingest_repository, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract AST-aware code chunks from a repo.")
    parser.add_argument("--repo-path", required=True, help="Path to the source repository.")
    parser.add_argument("--repo-name", help="Name to store in chunk metadata.")
    parser.add_argument(
        "--out",
        default="data/chunks/chunks.jsonl",
        help="Path to write JSONL chunks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_path = Path(args.repo_path)
    repo_name = args.repo_name or repo_path.resolve().name

    result = ingest_repository(repo_path=repo_path, repo_name=repo_name)
    write_jsonl(args.out, result.chunks)

    print(f"Scanned files: {result.files_scanned}")
    print(f"Extracted chunks: {len(result.chunks)}")
    print(f"Output: {args.out}")

    if result.errors:
        print(f"Parse/read warnings: {len(result.errors)}")
        for error in result.errors:
            print(f"- {error.path}: {error.error}")


if __name__ == "__main__":
    main()
