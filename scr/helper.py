import os
import sys
from pathlib import Path


if "__file__" in globals() or (
    sys.argv and ("models.py" in sys.argv[0] or "helper.py" in sys.argv[0])
):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path(os.getcwd()).resolve()
    if PROJECT_ROOT.name == "notebooks" or "notebooks" in PROJECT_ROOT.parts[-1]:
        PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_PATH = PROJECT_ROOT / "outputs"
MODELS_PATH = PROJECT_ROOT / "models"


def resolve_path(path: str) -> Path:
    target_path = PROJECT_ROOT / path

    if "models" in target_path.parts:
        if target_path.suffix not in [".pt", ".pth"] and not target_path.is_dir():
            target_path = target_path.with_suffix(".pt")

    return target_path
