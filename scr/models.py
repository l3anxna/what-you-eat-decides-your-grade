from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from helper import resolve_path


class AbstractModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

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
