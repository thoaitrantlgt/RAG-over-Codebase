import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    current: float
    delta: float
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalComparison:
    passed: bool
    deltas: list[MetricDelta]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "deltas": [delta.to_dict() for delta in self.deltas],
        }


def compare_reports(
    *,
    baseline_path: str | Path,
    current_path: str | Path,
    metrics: list[str],
    max_drop: float,
    section: str = "metrics",
) -> EvalComparison:
    baseline = load_metrics(baseline_path, section=section)
    current = load_metrics(current_path, section=section)
    deltas: list[MetricDelta] = []

    for metric in metrics:
        if metric not in baseline:
            raise ValueError(f"Metric {metric!r} missing from baseline report")
        if metric not in current:
            raise ValueError(f"Metric {metric!r} missing from current report")
        baseline_value = float(baseline[metric])
        current_value = float(current[metric])
        delta = current_value - baseline_value
        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=baseline_value,
                current=current_value,
                delta=delta,
                threshold=-abs(max_drop),
                passed=delta >= -abs(max_drop),
            )
        )

    return EvalComparison(
        passed=all(delta.passed for delta in deltas),
        deltas=deltas,
    )


def load_metrics(path: str | Path, *, section: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if section:
        metrics = data.get(section)
        if not isinstance(metrics, dict):
            raise ValueError(f"Report {path} has no metrics section {section!r}")
        return metrics
    return data
