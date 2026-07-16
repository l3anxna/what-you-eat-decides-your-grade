import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature import TARGET_COL, clean_data, get_train_val_data
from helper import OUTPUT_PATH
from pipeline import build_model_factories, compute_shap_importances
from sklearn.metrics import mean_absolute_error

plt.style.use("ggplot")

SHAP_THRESHOLD = 0.01
STEP = 10
N_REPEATS = 5
RANDOM_STATE = 42

BASE_MODEL_NAMES = ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "Neural Net"]


def training_sizes(max_size: int, step: int = STEP) -> list[int]:
    """Multiples of `step` from `step` up to (and always ending at) `max_size`."""
    sizes = list(range(step, max_size, step))
    sizes.append(max_size)

    return sizes


def compute_learning_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    input_dim: int,
    n_repeats: int = N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """For each training-set size (multiples of `STEP`, capped at the full training pool),
    repeatedly draws a random subsample of that size (no replacement), refits every plain
    AND bagged model on it, and scores it against both the subsample itself (train MAE)
    and the same fixed validation set (val MAE). Comparing the two is the point: a
    validation curve that plateaus while the train/val gap stays wide means the model
    would likely improve with more data (high variance); a plateau where train MAE is
    already about as bad as val MAE means more data probably wouldn't help (high bias).

    Repeats are averaged so one unlucky/lucky small subsample doesn't dominate a point --
    skipped at the max size, where every "resample" would just be the same rows reordered.
    """
    rng = np.random.RandomState(random_state)
    sizes = training_sizes(len(X_train))

    plain_factories = build_model_factories(input_dim=input_dim)
    bagged_factories = build_model_factories(input_dim=input_dim, bagging=True)
    factories = {**plain_factories, **bagged_factories}

    rows = []
    for size in sizes:
        repeats = 1 if size == len(X_train) else n_repeats

        for name, factory in factories.items():
            train_scores, val_scores = [], []

            for _ in range(repeats):
                idx = rng.choice(len(X_train), size=size, replace=False)
                X_sub, y_sub = X_train.iloc[idx], y_train.iloc[idx]

                model = factory().fit(X_sub, y_sub)
                train_scores.append(mean_absolute_error(y_sub, model.predict(X_sub)))
                val_scores.append(mean_absolute_error(y_val, model.predict(X_val)))

            rows.append({
                "model": name,
                "train_size": size,
                "train_MAE_mean": np.mean(train_scores),
                "train_MAE_std": np.std(train_scores),
                "val_MAE_mean": np.mean(val_scores),
                "val_MAE_std": np.std(val_scores),
            })

    return pd.DataFrame(rows)


def plot_model_learning_curves(curve_df: pd.DataFrame, output_dir=OUTPUT_PATH) -> list:
    """One PNG per base model, each with two side-by-side panels (plain vs. bagged),
    each panel showing train MAE and validation MAE vs. training set size on a linear
    (non-log) x-axis."""
    output_paths = []

    for base_name in BASE_MODEL_NAMES:
        bagged_name = f"{base_name} (Bagged)"

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

        for ax, name in zip(axes, [base_name, bagged_name]):
            sub = curve_df[curve_df["model"] == name].sort_values("train_size")

            ax.errorbar(
                sub["train_size"], sub["train_MAE_mean"], yerr=sub["train_MAE_std"],
                marker="o", capsize=3, label="Train MAE",
            )
            ax.errorbar(
                sub["train_size"], sub["val_MAE_mean"], yerr=sub["val_MAE_std"],
                marker="o", capsize=3, label="Validation MAE",
            )

            ax.set_title(name)
            ax.set_xlabel("Training set size (rows)")
            ax.legend()

        axes[0].set_ylabel("MAE (lower is better)")
        fig.suptitle(f"Learning Curve -- {base_name}: Plain vs Bagged")
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        slug = base_name.lower().replace(" ", "_")
        output_path = output_dir / f"learning_curve_{slug}.png"
        plt.savefig(output_path)
        plt.close(fig)

        output_paths.append(output_path)

    return output_paths


def main():
    df = clean_data()
    X_full = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    importances = compute_shap_importances(X_full, y)
    mask = importances[(importances > SHAP_THRESHOLD).all(axis=1)].index.tolist()

    X_train, X_val, y_train, y_val = get_train_val_data(mask=mask)

    curve_df = compute_learning_curve(X_train, y_train, X_val, y_val, input_dim=len(mask))
    print(curve_df.to_string(index=False))

    output_paths = plot_model_learning_curves(curve_df)
    print("\nLearning curves saved to:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
