import json
import os
from typing import Any, Callable

# Must run before `import torch` (pulled in below via `models`): torch and xgboost
# crash the process (SIGSEGV) from a native OpenMP thread-pool conflict otherwise.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from feature import TARGET_COL, clean_data, get_train_val_data, mask_features  # noqa: E402
from helper import MODELS_PATH  # noqa: E402
from models import BaggingEnsemble, GPARegressorNN, SklearnModel  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402
from sklearn.utils import resample  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

SHAP_THRESHOLD = 0.01
FEATURE_MASK_PATH = MODELS_PATH / "feature_mask.json"

# Bootstrap SHAP-selection stability (see `bootstrap_shap_selection_stability`): how many
# resamples to run, and what fraction of them a feature must be agreed-upon-important in
# to survive into the final feature mask.
BOOTSTRAP_ITERS = 100
SELECTION_FREQ_THRESHOLD = 0.5

# How many bootstrap-resampled members each bagged model factory trains. Cheap to raise
# since the dataset is tiny (~100 rows), but training time scales linearly with it.
BAGGING_ESTIMATORS = 15

MODEL_SLUGS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "XGBoost": "xgboost",
    "Neural Net": "neural_net",
    "Linear Regression (Bagged)": "linear_regression_bagged",
    "Random Forest (Bagged)": "random_forest_bagged",
    "Gradient Boosting (Bagged)": "gradient_boosting_bagged",
    "XGBoost (Bagged)": "xgboost_bagged",
    "Neural Net (Bagged)": "neural_net_bagged",
}


class ShapFeatureSelector:
    """Cross-model SHAP feature agreement, generalized from the EDA notebook's
    RandomForest/GradientBoosting comparison to an arbitrary set of already-fitted,
    tree-based (shap.TreeExplainer-compatible) sklearn-API regressors."""

    def __init__(self, models: dict[str, Any]):
        self.models = models

    def compute_importances(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, one column per model."""
        importances = {}

        for name, model in self.models.items():
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            importances[name] = pd.Series(
                np.abs(shap_values).mean(axis=0), index=X.columns
            )

        return pd.DataFrame(importances)

    def select_agreed_features(
        self, X: pd.DataFrame, threshold: float = 0.01
    ) -> list[str]:
        """Features whose mean absolute SHAP value exceeds `threshold` in every model."""
        importances = self.compute_importances(X)

        agreed = importances[(importances > threshold).all(axis=1)]

        return agreed.index.tolist()


def bootstrap_shap_selection_stability(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = SHAP_THRESHOLD,
    n_boot: int = BOOTSTRAP_ITERS,
    random_state: int = 42,
) -> pd.Series:
    """Ports the EDA notebook's bootstrap SHAP-stability check: resample (X, y) with
    replacement `n_boot` times, refit fresh RandomForest/GradientBoosting on an 80/20
    split of each resample, recompute SHAP importances, and record which features clear
    `threshold` mean-|SHAP| for *both* models on that resample.

    A single train/test split's SHAP-agreement set is unreliable on a dataset this small
    (~100 rows) -- it's fitting split-specific noise as much as signal. Returns, per
    feature, the fraction of the `n_boot` resamples in which it was selected; a feature
    worth trusting should show up most of the time, not just once.
    """
    rng = np.random.RandomState(random_state)
    selection_counts = pd.Series(0, index=X.columns, dtype=float)

    for _ in range(n_boot):
        X_boot, y_boot = resample(X, y, random_state=rng.randint(0, 1_000_000))
        X_train, X_val, y_train, y_val = train_test_split(
            X_boot, y_boot, test_size=0.2, random_state=rng.randint(0, 1_000_000)
        )

        rf_boot = RandomForestRegressor(random_state=0).fit(X_train, y_train)
        gb_boot = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)

        rf_importance = np.abs(shap.TreeExplainer(rf_boot).shap_values(X_val)).mean(
            axis=0
        )
        gb_importance = np.abs(shap.TreeExplainer(gb_boot).shap_values(X_val)).mean(
            axis=0
        )

        rf_selected = set(X_val.columns[rf_importance > threshold])
        gb_selected = set(X_val.columns[gb_importance > threshold])

        selection_counts[list(rf_selected & gb_selected)] += 1

    return (selection_counts / n_boot).sort_values(ascending=False)


def compare_models(
    model_factories: dict[str, Callable[[], Any]],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Single train/val split comparison: fits each model once, scores it on the held-out split."""
    rows = []

    for name, factory in model_factories.items():
        model = factory().fit(X_train, y_train)
        preds = model.predict(X_val)

        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_val, preds),
                "R2": r2_score(y_val, preds),
            }
        )

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def cross_validate_models(
    model_factories: dict[str, Callable[[], Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """K-fold comparison. A fresh model is built per fold (via `model_factories`) so every
    fold trains from scratch instead of continuing an already-fitted model's weights."""
    fold_indices = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X)
    )

    rows = []
    for name, factory in model_factories.items():
        mae_scores, r2_scores = [], []

        for train_idx, val_idx in fold_indices:
            model = factory().fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])

            mae_scores.append(mean_absolute_error(y.iloc[val_idx], preds))
            r2_scores.append(r2_score(y.iloc[val_idx], preds))

        rows.append(
            {
                "model": name,
                "MAE_mean": np.mean(mae_scores),
                "MAE_std": np.std(mae_scores),
                "R2_mean": np.mean(r2_scores),
                "R2_std": np.std(r2_scores),
            }
        )

    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def compute_shap_importances(X, y):
    rf = RandomForestRegressor(random_state=42).fit(X, y)
    gb = GradientBoostingRegressor(random_state=42).fit(X, y)

    return ShapFeatureSelector(
        {"RandomForest": rf, "GradientBoosting": gb}
    ).compute_importances(X)


def build_model_factories(
    input_dim: int, bagging: bool = False, n_estimators: int = BAGGING_ESTIMATORS
) -> dict:
    """`bagging=False` (default) returns the plain model factories. `bagging=True` wraps
    each of those same factories in a `BaggingEnsemble`, so every model also gets a
    bootstrap-aggregated counterpart to compare against -- useful on a dataset this small,
    where reducing single-fit variance often matters more than the base estimator choice."""
    base_factories = {
        "Linear Regression": lambda: SklearnModel(LinearRegression()),
        "Random Forest": lambda: SklearnModel(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        ),
        "Gradient Boosting": lambda: SklearnModel(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        ),
        # n_jobs=1: xgboost's own OpenMP threads crash (SIGSEGV) when mixed with
        # sklearn's native tree code in the same process on macOS otherwise.
        "XGBoost": lambda: SklearnModel(
            XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)
        ),
        "Neural Net": lambda: GPARegressorNN(input_dim=input_dim),
    }

    if not bagging:
        return base_factories

    return {
        f"{name} (Bagged)": (
            lambda factory=factory: BaggingEnsemble(factory, n_estimators=n_estimators)
        )
        for name, factory in base_factories.items()
    }


def run_training_comparison(
    threshold: float = SHAP_THRESHOLD,
    selection_freq_threshold: float = SELECTION_FREQ_THRESHOLD,
    n_boot: int = BOOTSTRAP_ITERS,
    n_bagging_estimators: int = BAGGING_ESTIMATORS,
) -> dict:
    """Replicates the notebook's modeling flow, upgraded with two bootstrap-based steps
    that matter more the smaller the dataset gets (~100 rows here):

    1. Feature mask: instead of trusting one 80/20 split's SHAP agreement, features are
       kept only if they clear the SHAP threshold for both RF and GB across a majority of
       `n_boot` bootstrap resamples (`bootstrap_shap_selection_stability`).
    2. Models: every model is compared both plain and as a `BaggingEnsemble` (bootstrap-
       aggregated) counterpart, so the model-comparison tables and the selected
       `best_model` can end up being either, whichever actually generalizes better.
    """
    df = clean_data()
    X_full = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    importances = compute_shap_importances(X_full, y)

    selection_freq = bootstrap_shap_selection_stability(
        X_full, y, threshold=threshold, n_boot=n_boot
    )
    mask = selection_freq[selection_freq >= selection_freq_threshold].index.tolist()
    if not mask:
        # Nothing cleared the stability bar -- fall back to the single-split agreement
        # rather than silently training on zero features.
        mask = importances[(importances > threshold).all(axis=1)].index.tolist()

    X_masked = mask_features(X_full, mask)
    X_train, X_val, y_train, y_val = get_train_val_data(mask=mask)

    plain_factories = build_model_factories(input_dim=len(mask))
    bagged_factories = build_model_factories(
        input_dim=len(mask), bagging=True, n_estimators=n_bagging_estimators
    )
    factories = {**plain_factories, **bagged_factories}

    single_split_df = compare_models(factories, X_train, X_val, y_train, y_val)
    cv_df = cross_validate_models(factories, X_masked, y)

    best_name = cv_df.iloc[0]["model"]
    best_model = factories[best_name]().fit(X_masked, y)

    return {
        "mask": mask,
        "selection_freq": selection_freq,
        "importances": importances,
        "single_split": single_split_df,
        "cv": cv_df,
        "best_model": best_model,
        "best_model_name": best_name,
    }


def train_final_models(
    mask: list[str], n_bagging_estimators: int = BAGGING_ESTIMATORS
) -> dict[str, Any]:
    """Fits every plain and bagged model from `build_model_factories` on the full (masked)
    dataset -- these are the ones that get persisted to disk for the CLI to load, so any
    of them (not just whichever CV happened to pick as `best_model`) can be loaded later."""
    df = clean_data()
    X = mask_features(df.drop(columns=[TARGET_COL]), mask)
    y = df[TARGET_COL]

    plain_factories = build_model_factories(input_dim=len(mask))
    bagged_factories = build_model_factories(
        input_dim=len(mask), bagging=True, n_estimators=n_bagging_estimators
    )
    factories = {**plain_factories, **bagged_factories}

    return {name: factory().fit(X, y) for name, factory in factories.items()}


def save_all_models(models: dict[str, Any], mask: list[str]):
    for name, model in models.items():
        model.save_model(MODEL_SLUGS[name])

    FEATURE_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_MASK_PATH.write_text(
        json.dumps({"target": TARGET_COL, "features": mask}, indent=2)
    )

    print(f"Saved {len(models)} models + feature mask to: {MODELS_PATH}")


def pipeline():
    results = run_training_comparison()

    print(
        f"Bootstrap-stable SHAP-selected features ({len(results['mask'])}): {results['mask']}"
    )
    print("\nSelection frequency across bootstrap resamples (top 15):")
    print(results["selection_freq"].head(15).to_string())
    print("\nSingle 80/20 split comparison:")
    print(results["single_split"].to_string(index=False))
    print("\n5-fold CV comparison:")
    print(results["cv"].to_string(index=False))
    print(f"\nBest model by CV MAE: {results['best_model_name']}")

    final_models = train_final_models(results["mask"])
    save_all_models(final_models, results["mask"])

    return results["best_model"]


if __name__ == "__main__":
    pipeline()
