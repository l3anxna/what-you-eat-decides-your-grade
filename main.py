import argparse
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
from feature import TARGET_COL, categorical_choices, clean_data, mask_features  # noqa: E402
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

# "_coded" features in the current mask, mapped to their paired raw free-text column --
# used to show real string choices instead of raw integer codes. Update if pipeline.py
# is retrained onto a different feature mask that includes/excludes different columns.
CATEGORICAL_FEATURES = {
    "fav_cuisine_coded": "fav_cuisine",
    "eating_changes_coded1": "eating_changes",
}

MAX_CHOICE_LABEL_LEN = 48

# Human-readable context for each feature, shown above its prompt. The dataset ships
# with no codebook, so entries marked "best guess" are inferred from the column name and
# the shape of its values, not confirmed against original survey wording -- treat them
# as a steer for how to answer, not a certainty.
FEATURE_INFO = {
    "parents_cook": (
        "Best guess: how often your parents cooked meals at home growing up. \n"
        "where lower = more often (e.g. 1 = most days, 5 = rarely/never)."
    ),
    "grade_level": "Your class year in college: (1) 1st year, (2) 2nd year, (3) 3rd year, (4) 4th year.",
    "weight": "Self-reported body weight, in pounds.",
    "father_education": (
        "Best guess: your father's highest level of education, on an ordinal scale \n"
        "(e.g. 1 = less than high school ... 5 = graduate degree)."
    ),
    "fav_cuisine_coded": "Your favorite cuisine.",
    "eating_changes_coded1": "How your eating habits have changed since starting college.",
}


def load_feature_metadata() -> dict:
    if not FEATURE_MASK_PATH.exists():
        raise FileNotFoundError(
            f"No trained models found at {MODELS_PATH}. "
            "Run `uv run --group ml python scr/pipeline.py` first to train and save them."
        )

    return json.loads(FEATURE_MASK_PATH.read_text())


def load_saved_model(slug: str, input_dim: int):
    if slug.endswith("_bagged"):
        # The factory is only needed by BaggingEnsemble.fit() to train fresh members --
        # load_model() replaces estimators_/n_estimators/random_state from disk directly.
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


def run_test_prediction():
    """--test-prediction: the original dev/test interface -- pick any of the 10 saved
    models and predict on a handful of pre-generated sample rows."""
    try:
        meta = load_feature_metadata()
    except FileNotFoundError as e:
        print(e)
        return

    mask = meta["features"]

    slug = prompt_model_choice()
    if slug is None:
        return

    try:
        model = load_saved_model(slug, input_dim=len(mask))
    except ModuleNotFoundError as e:
        print(
            f"{e}\nThis model needs the `ml` dependency group. "
            "Rerun with: uv run --group ml python main.py --test-prediction"
        )
        return

    sample = load_sample_data()
    X = mask_features(sample.drop(columns=[TARGET_COL]), mask)
    preds = model.predict(X)

    print("\nPredictions:")
    for i, (actual, pred) in enumerate(zip(sample[TARGET_COL], preds)):
        print(f"  row {i}: actual GPA={actual:.2f}  predicted GPA={pred:.2f}")


def numeric_bounds(feature: str) -> tuple[float, float, bool]:
    """(min, max, is_integer) for a numeric feature, derived from the cleaned dataset."""
    col = clean_data()[feature].dropna()
    is_integer = bool((col % 1 == 0).all())

    return float(col.min()), float(col.max()), is_integer


def prompt_numeric(index: int, total: int, feature: str) -> float:
    low, high, is_integer = numeric_bounds(feature)
    type_label = "int" if is_integer else "float"
    bounds_label = f"{int(low)}-{int(high)}" if is_integer else f"{low}-{high}"

    print(f"({index}/{total}) {feature} -- {type_label}, range {bounds_label}")
    if feature in FEATURE_INFO:
        print(f"    {FEATURE_INFO[feature]}")

    while True:
        raw = input("> ").strip()

        if not raw:
            print("A value is required.")
            continue

        try:
            value = float(raw)
        except ValueError:
            print(f"Enter a number in range {bounds_label}.")
            continue

        if is_integer and not value.is_integer():
            print(f"Enter a whole number in range {bounds_label}.")
            continue

        if not (low <= value <= high):
            print(f"Out of range -- enter a value between {bounds_label}.")
            continue

        return value


def prompt_categorical(index: int, total: int, feature: str, raw_text_col: str) -> int:
    choices = categorical_choices(feature, raw_text_col)
    ordered_codes = sorted(choices)

    print(f"({index}/{total}) {feature} -- choose one")
    if feature in FEATURE_INFO:
        print(f"    {FEATURE_INFO[feature]}")
    for i, code in enumerate(ordered_codes, start=1):
        label = choices[code]
        if len(label) > MAX_CHOICE_LABEL_LEN:
            label = label[: MAX_CHOICE_LABEL_LEN - 3] + "..."
        print(f"    {i}) {label}")

    while True:
        raw = input("  > ").strip()

        if not raw:
            print("A choice is required.")
            continue

        if raw.isdigit() and 1 <= int(raw) <= len(ordered_codes):
            return ordered_codes[int(raw) - 1]

        print(f"Enter a number 1-{len(ordered_codes)}.")


def collect_feature_values(mask: list[str]) -> dict:
    values = {}

    for i, feature in enumerate(mask, start=1):
        if feature in CATEGORICAL_FEATURES:
            values[feature] = prompt_categorical(i, len(mask), feature, CATEGORICAL_FEATURES[feature])
        else:
            values[feature] = prompt_numeric(i, len(mask), feature)

    return values


def run_interactive_predictor():
    """Default entry point: ask the user for their own answers to the SHAP-selected
    features, then predict GPA with the best model from the last training run."""
    try:
        meta = load_feature_metadata()
    except FileNotFoundError as e:
        print(e)
        return

    mask = meta["features"]
    best_slug = meta.get("best_model")
    if not best_slug:
        print(
            "No best-model recorded in feature_mask.json -- retrain with "
            "`uv run --group ml python scr/pipeline.py` to regenerate it."
        )
        return

    print("Answer the following to predict your GPA:\n")

    try:
        values = collect_feature_values(mask)
    except EOFError:
        print("\nNo input received -- exiting.")
        return

    try:
        model = load_saved_model(best_slug, input_dim=len(mask))
    except ModuleNotFoundError as e:
        print(f"{e}\nThis model needs the `ml` dependency group. Rerun with: uv run --group ml python main.py")
        return

    X = pd.DataFrame([values])[mask]
    pred = model.predict(X)[0]

    print(f"\nPredicted GPA: {pred:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Predict GPA from food & lifestyle habits.")
    parser.add_argument(
        "--test-prediction",
        action="store_true",
        help="Use the old 10-model picker against pre-generated sample data (dev/testing).",
    )
    args = parser.parse_args()

    if args.test_prediction:
        run_test_prediction()
    else:
        run_interactive_predictor()


if __name__ == "__main__":
    main()
