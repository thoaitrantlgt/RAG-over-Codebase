import json
import tempfile
import unittest
from pathlib import Path

from packages.evals.compare import compare_reports


class EvalCompareTests(unittest.TestCase):
    def test_compare_reports_passes_within_drop_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.json"
            current = root / "current.json"
            write_report(baseline, {"mrr_at_10": 0.50, "ndcg_at_10": 0.60})
            write_report(current, {"mrr_at_10": 0.49, "ndcg_at_10": 0.59})

            result = compare_reports(
                baseline_path=baseline,
                current_path=current,
                metrics=["mrr_at_10", "ndcg_at_10"],
                max_drop=0.02,
            )

        self.assertTrue(result.passed)

    def test_compare_reports_fails_when_metric_drops_too_much(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.json"
            current = root / "current.json"
            write_report(baseline, {"mrr_at_10": 0.50})
            write_report(current, {"mrr_at_10": 0.40})

            result = compare_reports(
                baseline_path=baseline,
                current_path=current,
                metrics=["mrr_at_10"],
                max_drop=0.02,
            )

        self.assertFalse(result.passed)
        self.assertLess(result.deltas[0].delta, 0)


def write_report(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
