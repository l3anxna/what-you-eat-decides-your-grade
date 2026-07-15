from abc import ABC, abstractmethod
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from helper import resolve_path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


class NNAbstractModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _to_tensor(data) -> torch.Tensor:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_numpy()

        return torch.as_tensor(np.asarray(data).copy(), dtype=torch.float32)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ) -> "NNAbstractModel":
        """Generic MSE-regression training loop shared by every NNAbstractModel subclass."""
        self.train()

        X_t = self._to_tensor(X)
        y_t = self._to_tensor(y)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.forward(X_t), y_t)
            loss.backward()
            optimizer.step()

        self.eval()

        return self

    @property
    def device(self) -> torch.device:
        """Dynamically identifies what hardware this instance is using."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_model(self, name: str):
        """Saves model weights into the absolute ./models/ directory path."""
        full_path = resolve_path(f"models/{name}")
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.state_dict(), full_path)
        print(f"Model weights successfully saved to: {full_path}")

        return

    def load_model(self, name: str):
        """Loads weights safely across Google Colab (T4/CPU) and GitHub CI (ARM CPU)."""
        full_path = resolve_path(f"models/{name}")
        
        if not full_path.exists():
            raise FileNotFoundError(f"No model file found at designated path: {full_path}")

        state_dict = torch.load(full_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
        self.eval()
        
        print(f"Model weights successfully loaded on [{self.device}] from: {full_path}")

        return


class SklearnAbstractModel(ABC):
    def __init__(self):
        self.model = self.build_model()

    @abstractmethod
    def build_model(self) -> Any:
        """Constructs and returns the underlying (unfitted) sklearn-API estimator."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnAbstractModel":
        self.model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save_model(self, name: str):
        """Saves the fitted estimator into the absolute ./models/ directory path."""
        full_path = resolve_path(f"models/{name}").with_suffix(".joblib")

        full_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, full_path)
        print(f"Model successfully saved to: {full_path}")

        return

    def load_model(self, name: str):
        """Loads a previously saved fitted estimator."""
        full_path = resolve_path(f"models/{name}").with_suffix(".joblib")

        if not full_path.exists():
            raise FileNotFoundError(f"No model file found at designated path: {full_path}")

        self.model = joblib.load(full_path)
        print(f"Model successfully loaded from: {full_path}")

        return


class SklearnModel(SklearnAbstractModel):
    """Thin SklearnAbstractModel wrapper around any already-constructed,
    unfitted sklearn/xgboost-API estimator (e.g. SklearnModel(LinearRegression()))."""

    def __init__(self, estimator: Any):
        self._estimator = estimator

        super().__init__()

    def build_model(self) -> Any:
        return self._estimator


class GPARegressorNN(NNAbstractModel):
    """Small MLP counterpart to the sklearn/xgboost regressors, kept deliberately
    shallow (single hidden layer + dropout) given the dataset only has ~100 rows."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.3, seed: int = 42):
        super().__init__()

        torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.eval()

        with torch.no_grad():
            return self.forward(self._to_tensor(X)).cpu().numpy()


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

        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_val, preds),
            "R2": r2_score(y_val, preds),
        })

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
    fold_indices = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X))

    rows = []
    for name, factory in model_factories.items():
        mae_scores, r2_scores = [], []

        for train_idx, val_idx in fold_indices:
            model = factory().fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])

            mae_scores.append(mean_absolute_error(y.iloc[val_idx], preds))
            r2_scores.append(r2_score(y.iloc[val_idx], preds))

        rows.append({
            "model": name,
            "MAE_mean": np.mean(mae_scores),
            "MAE_std": np.std(mae_scores),
            "R2_mean": np.mean(r2_scores),
            "R2_std": np.std(r2_scores),
        })

    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


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

    def select_agreed_features(self, X: pd.DataFrame, threshold: float = 0.01) -> list[str]:
        """Features whose mean absolute SHAP value exceeds `threshold` in every model."""
        importances = self.compute_importances(X)

        agreed = importances[(importances > threshold).all(axis=1)]

        return agreed.index.tolist()
