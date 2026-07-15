import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.parametrize("choice", ["1", "2", "3", "4", "5"])
def test_model_selection_runs(choice):
    """Smoke test: each menu choice should run end-to-end without crashing.
    Prediction accuracy is not the concern here, just that it completes."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py")],
        input=f"{choice}\n",
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Predictions:" in result.stdout
