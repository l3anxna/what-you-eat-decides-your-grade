import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from feature import TARGET_COL, clean_data, load_raw_data
from helper import OUTPUT_PATH
from pipeline import run_training_comparison
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

plt.style.use("ggplot")


def plot_missing_values(
    df: pd.DataFrame, output_path=OUTPUT_PATH / "missing_values.png"
):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(12, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_correlation_heatmap(
    df: pd.DataFrame, output_path=OUTPUT_PATH / "correlation_heatmap.png"
):
    corr = df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_correlation_with_target(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    output_path=OUTPUT_PATH / "correlation_with_gpa.png",
):
    corr_target = df.corr()[target].drop(target).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    corr_target.plot(kind="barh")
    plt.title(f"Correlation with {target}")
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_shap_importance(
    importances: pd.DataFrame,
    threshold: float = 0.01,
    output_path=OUTPUT_PATH / "shap_importance.png",
):
    agreed = importances[(importances > threshold).all(axis=1)]
    if agreed.empty:
        agreed = importances.loc[
            importances.mean(axis=1).sort_values(ascending=False).head(10).index
        ]

    agreed = agreed.sort_values(by=agreed.columns[0], ascending=False)
    plot_df = agreed.reset_index(names="feature").melt(
        id_vars="feature", var_name="model", value_name="mean_abs_shap"
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(x="mean_abs_shap", y="feature", hue="model", data=plot_df)
    plt.title("Mean Absolute SHAP Value by Model")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_selection_stability(
    selection_freq: pd.Series,
    threshold: float = 0.5,
    output_path=OUTPUT_PATH / "shap_selection_stability.png",
):
    """Bars showing what fraction of bootstrap resamples each feature cleared the SHAP
    agreement threshold in. Features above `threshold` (marked with a reference line) are
    the ones that made it into the final feature mask."""
    top = selection_freq.head(20).sort_values()

    plt.figure(figsize=(10, 8))
    colors = ["#55A868" if v >= threshold else "#C44E52" for v in top.values]
    top.plot(kind="barh", color=colors)
    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"threshold ({threshold})",
    )
    plt.title("Bootstrap SHAP-Selection Frequency by Feature")
    plt.xlabel("Fraction of Bootstrap Resamples Selected In")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_model_comparison(
    comparison: pd.DataFrame, output_path=OUTPUT_PATH / "model_comparison.png"
):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    by_mae = comparison.sort_values("MAE")
    axes[0].barh(by_mae["model"], by_mae["MAE"])
    axes[0].set_title("MAE by Model (80/20 split)")
    axes[0].set_xlabel("MAE (lower is better)")
    axes[0].invert_yaxis()

    by_r2 = comparison.sort_values("R2", ascending=False)
    axes[1].barh(by_r2["model"], by_r2["R2"])
    axes[1].set_title("R² by Model (80/20 split)")
    axes[1].set_xlabel("R² (higher is better)")
    axes[1].invert_yaxis()

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_cv_comparison(
    cv_comparison: pd.DataFrame, output_path=OUTPUT_PATH / "cv_comparison.png"
):
    by_mae = cv_comparison.sort_values("MAE_mean")

    plt.figure(figsize=(10, 6))
    plt.barh(by_mae["model"], by_mae["MAE_mean"], xerr=by_mae["MAE_std"], capsize=4)
    plt.title("5-Fold CV MAE by Model (mean ± std)")
    plt.xlabel("MAE (lower is better)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def run_anomaly_detection(X: pd.DataFrame) -> pd.Series:
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_pred = iso.fit_predict(X)

    lof = LocalOutlierFactor(n_neighbors=15, contamination=0.1)
    lof_pred = lof.fit_predict(X)

    flagged_by = (iso_pred == -1).astype(int) + (lof_pred == -1).astype(int)

    return pd.Series(flagged_by, index=X.index, name="flagged_by")


def plot_anomaly_agreement(
    flagged_by: pd.Series, output_path=OUTPUT_PATH / "anomaly_agreement.png"
):
    plt.figure(figsize=(10, 4))
    flagged_by.value_counts().sort_index().plot(
        kind="bar", color=["#4C72B0", "#DD8452", "#55A868"]
    )
    plt.title("Rows by Number of Anomaly-Detection Methods That Flagged Them")
    plt.xlabel("Number of Methods Agreeing (0-2)")
    plt.ylabel("Row Count")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def main():
    raw_df = load_raw_data()
    plot_missing_values(raw_df)

    df = clean_data()
    plot_correlation_heatmap(df)
    plot_correlation_with_target(df)

    X = df.drop(columns=[TARGET_COL])

    results = run_training_comparison()
    plot_shap_importance(results["importances"])
    plot_selection_stability(results["selection_freq"])
    plot_model_comparison(results["single_split"])
    plot_cv_comparison(results["cv"])

    flagged_by = run_anomaly_detection(X)
    plot_anomaly_agreement(flagged_by)

    print(f"Artefacts written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
