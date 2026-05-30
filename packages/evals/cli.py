import argparse
import json
import sys
from pathlib import Path

from .compare import compare_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation report utilities.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    compare = subcommands.add_parser("compare", help="Compare eval reports and fail on metric drops.")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--current", required=True)
    compare.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric to compare. Can be repeated.",
    )
    compare.add_argument("--max-drop", type=float, default=0.02)
    compare.add_argument("--section", default="metrics")
    compare.add_argument("--report-out")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    metrics = args.metric or ["mrr_at_10", "ndcg_at_10", "recall_at_10"]
    comparison = compare_reports(
        baseline_path=args.baseline,
        current_path=args.current,
        metrics=metrics,
        max_drop=args.max_drop,
        section=args.section,
    )
    payload = comparison.to_dict()
    if args.report_out:
        output = Path(args.report_out).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not comparison.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
