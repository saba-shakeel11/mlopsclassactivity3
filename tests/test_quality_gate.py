import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from quality_gate import main


def test_quality_gate_pass(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"accuracy": 0.95}), encoding="utf-8")

    # Should not raise SystemExit when accuracy meets the threshold
    main(str(metrics_file), 0.9)


def test_quality_gate_fail(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"accuracy": 0.85}), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(str(metrics_file), 0.9)

    assert "below threshold" in str(exc.value)
