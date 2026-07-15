import json
import os
import sys
from pathlib import Path

# Must run before `import torch` (pulled in below via `models`): torch and a joblib-loaded
# XGBoost model crash the process (SIGSEGV) from a native OpenMP thread-pool conflict otherwise.
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scr"))

import pandas as pd  # noqa: E402
from feature import TARGET_COL, mask_features  # noqa: E402
from helper import MODELS_PATH, PROJECT_ROOT  # noqa: E402
from models import BaggingEnsemble, GPARegressorNN, SklearnModel  # noqa: E402

FEATURE_MASK_PATH = MODELS_PATH / "feature_mask.json"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_input.csv"

MODEL_CHOICES = [
    ("Linear Regression", "linear_regression"),
    ("Random Forest", "random_forest"),
    ("Gradient Boosting", "gradient_boosting"),
    ("XGBoost", "xgboost"),
    ("Neural Net", "neural_net"),
    ("Linear Regression (Bagged)", "linear_regression_bagged"),
    ("Random Forest (Bagged)", "random_forest_bagged"),
    ("Gradient Boosting (Bagged)", "gradient_boosting_bagged"),
    ("XGBoost (Bagged)", "xgboost_bagged"),
    ("Neural Net (Bagged)", "neural_net_bagged"),
]


def load_feature_mask() -> list[str]:
    if not FEATURE_MASK_PATH.exists():
        raise FileNotFoundError(
            f"No trained models found at {MODELS_PATH}. "
            "Run `uv run --group ml python scr/pipeline.py` first to train and save them."
        )

    return json.loads(FEATURE_MASK_PATH.read_text())["features"]


def load_saved_model(slug: str, input_dim: int):
    if slug.endswith("_bagged"):
        # The factory is only needed by BaggingEnsemble.fit() to train fresh members --
        # load_model() replaces estimators_/n_estimators_/random_state from disk directly.
        model = BaggingEnsemble(factory=lambda: None)
    elif slug == "neural_net":
        model = GPARegressorNN(input_dim=input_dim)
    else:
        # The wrapped estimator is irrelevant here -- load_model() immediately
        # overwrites `self.model` with the joblib-deserialized, already-fitted one.
        model = SklearnModel(None)

    model.load_model(slug)

    return model


def load_sample_data() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        from sample_data import generate_sample_data

        return generate_sample_data()

    return pd.read_csv(SAMPLE_PATH)


def prompt_model_choice() -> str | None:
    print("Select a model:")
    for i, (label, _) in enumerate(MODEL_CHOICES, start=1):
        print(f"  {i}) {label}")

    while True:
        try:
            choice = input("> ").strip()
        except EOFError:
            print("\nNo input received -- exiting.")
            return None

        if choice.isdigit() and 1 <= int(choice) <= len(MODEL_CHOICES):
            return MODEL_CHOICES[int(choice) - 1][1]

        print(f"Invalid choice, enter a number 1-{len(MODEL_CHOICES)}.")


def main():
    try:
        mask = load_feature_mask()
    except FileNotFoundError as e:
        print(e)
        return

    slug = prompt_model_choice()
    if slug is None:
        return

    try:
        model = load_saved_model(slug, input_dim=len(mask))
    except ModuleNotFoundError as e:
        print(
            f"{e}\nThis model needs the `ml` dependency group. "
            "Rerun with: uv run --group ml python main.py"
        )
        return

    sample = load_sample_data()
    X = mask_features(sample.drop(columns=[TARGET_COL]), mask)
    preds = model.predict(X)

    print("\nPredictions:")
    for i, (actual, pred) in enumerate(zip(sample[TARGET_COL], preds)):
        print(f"  row {i}: actual GPA={actual:.2f}  predicted GPA={pred:.2f}")


if __name__ == "__main__":
    main()
