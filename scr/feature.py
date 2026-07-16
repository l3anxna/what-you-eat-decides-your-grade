import re

import numpy as np
import pandas as pd
from helper import resolve_path

DEFAULT_DATA_PATH = "data/food_coded.csv"
TARGET_COL = "GPA"


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(resolve_path(path or DEFAULT_DATA_PATH))


def convert_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["GPA"] = pd.to_numeric(df["GPA"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    return df


def normalize_text(text):
    if pd.isna(text):
        return text

    text = str(text).lower()
    text = text.strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].apply(normalize_text)

    return df


def clean_comfort_food_reasons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "comfort_food_reasons" not in df.columns:
        return df

    def bucket(text):
        if pd.isna(text):
            return text

        if "bored" in text:
            return "boredom"

        if "stress" in text:
            return "stress"

        if "sad" in text:
            return "sadness"

        if "happy" in text:
            return "happiness"

        if "depress" in text:
            return "depression"

        return text

    df["comfort_food_reasons"] = df["comfort_food_reasons"].apply(bucket)

    return df


def purge_unencodable_columns(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include="object").columns

    cols_to_purge = []
    for col in object_cols:
        string_values = df[col].dropna()

        is_free_text = string_values.apply(
            lambda x: isinstance(x, str) and x.count(" ") >= 3 and len(x) >= 25
        )

        if not string_values.empty and is_free_text.any():
            cols_to_purge.append(col)

    return df.drop(columns=cols_to_purge, errors="ignore")


def select_numeric_features(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if target not in numeric_cols:
        numeric_cols = [target] + numeric_cols

    return df[numeric_cols]


def impute_numeric_features(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    df = df.copy()

    feature_cols = [col for col in df.columns if col != target]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


def drop_missing_target(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    return df.dropna(subset=[target])


def clean_data(path: str | None = None) -> pd.DataFrame:
    df = load_raw_data(path)
    df = convert_numeric_types(df)
    df = normalize_text_columns(df)
    df = clean_comfort_food_reasons(df)
    df = purge_unencodable_columns(df)
    df = select_numeric_features(df)
    df = impute_numeric_features(df)
    df = drop_missing_target(df)

    return df


def mask_features(df: pd.DataFrame, mask: list[str]) -> pd.DataFrame:
    return df[mask]


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    shuffled_idx = rng.permutation(len(X))

    n_val = int(len(X) * val_size)
    val_idx, train_idx = shuffled_idx[:n_val], shuffled_idx[n_val:]

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    return X_train, X_val, y_train, y_val


def get_train_val_data(
    path: str | None = None,
    mask: list[str] | None = None,
    target: str = TARGET_COL,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = clean_data(path)

    X = df.drop(columns=[target])
    if mask is not None:
        X = mask_features(X, mask)

    y = df[target]

    return train_val_split(X, y, val_size, random_state)


def categorical_choices(coded_col: str, raw_text_col: str, path: str | None = None) -> dict[int, str]:
    """Derives {code: label} for a '_coded' categorical column by cross-referencing it
    against its paired raw free-text column in the original data. The label picked per
    code is its most frequent normalized text value (mode), tie-broken by shortest
    string -- mechanically "scraped" from the data rather than hand-authored, so it
    stays correct if the underlying CSV changes."""
    raw = load_raw_data(path)
    normalized = raw[raw_text_col].apply(normalize_text)

    choices = {}
    for code, group in normalized.groupby(raw[coded_col]):
        valid = group.dropna()
        if valid.empty:
            continue

        counts = valid.value_counts()
        top = counts[counts == counts.max()].index.tolist()
        choices[int(code)] = min(top, key=len)

    return choices
