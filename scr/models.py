import torch
import torch.nn as nn
from helper import resolve_path
from abc import ABC, abstractmethod


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
