from abc import ABC, abstractmethod
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from helper import resolve_path
from sklearn.utils import resample


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

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
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
            raise FileNotFoundError(
                f"No model file found at designated path: {full_path}"
            )

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
            raise FileNotFoundError(
                f"No model file found at designated path: {full_path}"
            )

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

    def __init__(
        self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.3, seed: int = 42
    ):
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


class BaggingEnsemble:
    """Bootstrap-aggregation wrapper: fits `n_estimators` independent copies of whatever
    `factory()` returns (any SklearnAbstractModel or NNAbstractModel instance -- both
    already expose a matching .fit(X, y) / .predict(X) interface) on a fresh row-resample
    (with replacement) of the training data each time, then averages the member
    predictions.

    Point of this on a ~100-row dataset: a single fit is highly sensitive to exactly which
    rows land in the training set. Averaging many resampled fits smooths that out and
    tends to reduce validation-set variance (and often MAE) versus any one fit, at the
    cost of `n_estimators`x the training time. Cheap here since the dataset is tiny.
    """

    def __init__(
        self, factory: Callable[[], Any], n_estimators: int = 15, random_state: int = 42
    ):
        self.factory = factory
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_: list[Any] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaggingEnsemble":
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            X_boot, y_boot = resample(X, y, random_state=rng.randint(0, 1_000_000))
            estimator = self.factory().fit(X_boot, y_boot)
            self.estimators_.append(estimator)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        member_preds = np.stack(
            [estimator.predict(X) for estimator in self.estimators_]
        )

        return member_preds.mean(axis=0)

    def save_model(self, name: str):
        """Persists just the fitted member estimators + metadata via joblib, rather than
        pickling `self` whole -- `self.factory` is typically a lambda (from
        `build_model_factories`), which pickle/joblib cannot serialize. The factory is
        only needed to produce *new* members during `fit`; a reloaded ensemble is for
        inference (`predict`) and doesn't need it."""
        full_path = resolve_path(f"models/{name}").with_suffix(".joblib")

        full_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "estimators_": self.estimators_,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        joblib.dump(payload, full_path)
        print(f"Bagging ensemble ({self.n_estimators} members) saved to: {full_path}")

        return

    def load_model(self, name: str):
        """Restores fitted members for inference. Note: `self.factory` is not restored
        (it wasn't saved -- see `save_model`), so calling `.fit()` again on a reloaded
        ensemble will fail unless a new `factory` is assigned first."""
        full_path = resolve_path(f"models/{name}").with_suffix(".joblib")

        if not full_path.exists():
            raise FileNotFoundError(
                f"No model file found at designated path: {full_path}"
            )

        payload = joblib.load(full_path)
        self.estimators_ = payload["estimators_"]
        self.n_estimators = payload["n_estimators"]
        self.random_state = payload["random_state"]
        print(f"Bagging ensemble loaded from: {full_path}")

        return
