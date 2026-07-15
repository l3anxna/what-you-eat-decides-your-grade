from feature import TARGET_COL, clean_data, get_train_val_data, mask_features
from models import GPARegressorNN, ShapFeatureSelector, SklearnModel, compare_models, cross_validate_models
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

SHAP_THRESHOLD = 0.01


def compute_shap_importances(X, y):
    rf = RandomForestRegressor(random_state=42).fit(X, y)
    gb = GradientBoostingRegressor(random_state=42).fit(X, y)

    return ShapFeatureSelector({"RandomForest": rf, "GradientBoosting": gb}).compute_importances(X)


def build_model_factories(input_dim: int) -> dict:
    return {
        "Linear Regression": lambda: SklearnModel(LinearRegression()),
        "Random Forest": lambda: SklearnModel(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        ),
        "Gradient Boosting": lambda: SklearnModel(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        # n_jobs=1: xgboost's own OpenMP threads crash (SIGSEGV) when mixed with
        # sklearn's native tree code in the same process on macOS otherwise.
        "XGBoost": lambda: SklearnModel(XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        "Neural Net": lambda: GPARegressorNN(input_dim=input_dim),
    }


def run_training_comparison(threshold: float = SHAP_THRESHOLD) -> dict:
    """Replicates the notebook's modeling flow: clean -> SHAP-agree on a feature mask ->
    train/compare every model (sklearn + NN) on a single split and via 5-fold CV."""
    df = clean_data()
    X_full = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    importances = compute_shap_importances(X_full, y)
    mask = importances[(importances > threshold).all(axis=1)].index.tolist()

    X_masked = mask_features(X_full, mask)
    X_train, X_val, y_train, y_val = get_train_val_data(mask=mask)

    factories = build_model_factories(input_dim=len(mask))

    single_split_df = compare_models(factories, X_train, X_val, y_train, y_val)
    cv_df = cross_validate_models(factories, X_masked, y)

    best_name = cv_df.iloc[0]["model"]
    best_model = factories[best_name]().fit(X_masked, y)

    return {
        "mask": mask,
        "importances": importances,
        "single_split": single_split_df,
        "cv": cv_df,
        "best_model": best_model,
        "best_model_name": best_name,
    }


def pipeline():
    results = run_training_comparison()

    print(f"SHAP-selected features ({len(results['mask'])}): {results['mask']}")
    print("\nSingle 80/20 split comparison:")
    print(results["single_split"].to_string(index=False))
    print("\n5-fold CV comparison:")
    print(results["cv"].to_string(index=False))
    print(f"\nBest model by CV MAE: {results['best_model_name']}")

    return results["best_model"]


if __name__ == "__main__":
    pipeline()
