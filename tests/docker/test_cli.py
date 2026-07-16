import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.parametrize("choice", [str(n) for n in range(1, 11)])
def test_model_selection_runs(choice):
    """Smoke test: each menu choice should run end-to-end without crashing.
    Prediction accuracy is not the concern here, just that it completes."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--test-prediction"],
        input=f"{choice}\n",
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Predictions:" in result.stdout


def test_interactive_predictor_runs():
    """Smoke test for the default (no-flag) entry point: answer all 6 prompts and
    confirm a prediction comes out the other end. Accuracy is not the concern."""
    # parents_cook, grade_level, weight, father_education, fav_cuisine choice, eating_changes choice
    answers = "1\n1\n150\n1\n1\n1\n"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py")],
        input=answers,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Predicted GPA:" in result.stdout
