import json
import sys
from pathlib import Path


def main(metrics_path: str, accuracy_threshold: float) -> None:
    path = Path(metrics_path)
    if not path.exists():
        raise SystemExit(f"Metrics file not found at {metrics_path}")

    with path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy")
    if accuracy is None:
        raise SystemExit("Accuracy metric missing from metrics file")

    if accuracy < accuracy_threshold:
        raise SystemExit(
            f"Accuracy {accuracy:.4f} is below threshold {accuracy_threshold:.4f}. Failing quality gate."
        )

    print(
        f"Quality gate passed. Accuracy {accuracy:.4f} is above threshold {accuracy_threshold:.4f}."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python quality_gate.py <metrics_path> <accuracy_threshold>")

    metrics_arg = sys.argv[1]
    try:
        threshold_arg = float(sys.argv[2])
    except ValueError as exc:
        raise SystemExit("Accuracy threshold must be a numeric value") from exc

    main(metrics_arg, threshold_arg)
