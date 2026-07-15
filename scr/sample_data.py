import pandas as pd
from feature import clean_data
from helper import PROJECT_ROOT

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_input.csv"


def generate_sample_data(n: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Draws a small, already-cleaned sample to demo inference with (same fully-numeric,
    modeling-ready shape `feature.clean_data()` produces -- nothing left to impute)."""
    df = clean_data()
    sample = df.sample(n=n, random_state=random_state).reset_index(drop=True)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)

    print(f"Saved {n}-row sample to: {SAMPLE_PATH}")

    return sample


if __name__ == "__main__":
    generate_sample_data()
