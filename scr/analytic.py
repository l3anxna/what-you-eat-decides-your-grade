import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from feature import TARGET_COL, clean_data, load_raw_data
from helper import OUTPUT_PATH
from models import ShapFeatureSelector
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor

plt.style.use("ggplot")


def plot_missing_values(df: pd.DataFrame, output_path=OUTPUT_PATH / "missing_values.png"):
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


def plot_correlation_heatmap(df: pd.DataFrame, output_path=OUTPUT_PATH / "correlation_heatmap.png"):
    corr = df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_correlation_with_target(
    df: pd.DataFrame, target: str = TARGET_COL, output_path=OUTPUT_PATH / "correlation_with_gpa.png"
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
    importances: pd.DataFrame, threshold: float = 0.01, output_path=OUTPUT_PATH / "shap_importance.png"
):
    agreed = importances[(importances > threshold).all(axis=1)]
    if agreed.empty:
        agreed = importances.loc[importances.mean(axis=1).sort_values(ascending=False).head(10).index]

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


def run_anomaly_detection(X: pd.DataFrame) -> pd.Series:
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_pred = iso.fit_predict(X)

    lof = LocalOutlierFactor(n_neighbors=15, contamination=0.1)
    lof_pred = lof.fit_predict(X)

    flagged_by = (iso_pred == -1).astype(int) + (lof_pred == -1).astype(int)

    return pd.Series(flagged_by, index=X.index, name="flagged_by")


def plot_anomaly_agreement(flagged_by: pd.Series, output_path=OUTPUT_PATH / "anomaly_agreement.png"):
    plt.figure(figsize=(10, 4))
    flagged_by.value_counts().sort_index().plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"])
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
    y = df[TARGET_COL]

    rf = RandomForestRegressor(random_state=42).fit(X, y)
    gb = GradientBoostingRegressor(random_state=42).fit(X, y)
    importances = ShapFeatureSelector({"RandomForest": rf, "GradientBoosting": gb}).compute_importances(X)
    plot_shap_importance(importances)

    flagged_by = run_anomaly_detection(X)
    plot_anomaly_agreement(flagged_by)

    print(f"Artefacts written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
