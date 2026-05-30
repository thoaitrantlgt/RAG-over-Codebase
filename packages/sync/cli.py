import argparse
import json
from pathlib import Path

from .git import changed_files
from .incremental import run_incremental_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 incremental sync utilities.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    changed = subcommands.add_parser("changed-files", help="List files changed between two git refs.")
    changed.add_argument("--repo-path", required=True)
    changed.add_argument("--base", required=True)
    changed.add_argument("--head", default="HEAD")
    changed.add_argument("--json", action="store_true")

    inc = subcommands.add_parser("incremental", help="Run incremental ingestion state diff.")
    inc.add_argument("--repo-path", required=True)
    inc.add_argument("--repo-name")
    inc.add_argument("--state", default="data/sync/sync_state.json")
    inc.add_argument("--changed-file", action="append", default=[])
    inc.add_argument("--changed-files")
    inc.add_argument("--base")
    inc.add_argument("--head", default="HEAD")
    inc.add_argument("--changed-chunks-out", default="data/sync/changed_chunks.jsonl")
    inc.add_argument("--full-chunks-out")
    inc.add_argument("--report-out", default="data/sync/incremental_report.json")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "changed-files":
        files = changed_files(repo_path=args.repo_path, base=args.base, head=args.head)
        if args.json:
            print(json.dumps({"changed_files": files}, ensure_ascii=False, indent=2))
        else:
            for path in files:
                print(path)
        return

    repo_path = Path(args.repo_path)
    repo_name = args.repo_name or repo_path.resolve().name
    selected_files = collect_changed_files(args)
    result = run_incremental_sync(
        repo_path=repo_path,
        repo_name=repo_name,
        state_path=args.state,
        changed_files=selected_files,
        changed_chunks_out=args.changed_chunks_out,
        full_chunks_out=args.full_chunks_out,
    )
    write_json(args.report_out, result.to_dict())
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def collect_changed_files(args) -> list[str] | None:
    files: list[str] = []
    files.extend(args.changed_file or [])

    if args.changed_files:
        for item in args.changed_files.split(","):
            clean = item.strip()
            if clean:
                files.append(clean)

    if args.base:
        files.extend(changed_files(repo_path=args.repo_path, base=args.base, head=args.head))

    return files or None


def write_json(path: str, payload: dict) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
