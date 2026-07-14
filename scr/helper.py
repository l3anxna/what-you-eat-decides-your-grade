import os
import sys
from pathlib import Path


def resolve_path(path: str) -> Path:
    """
    Resolves any relative path string against the project's root directory,
    ensuring consistent lookups across Colab, GitHub Actions, and local setups.
    """
    if "__file__" in globals() or "models.py" in sys.argv[0] or "helper.py" in sys.argv[0]:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(os.getcwd()).resolve()
        
        if project_root.name == "notebooks" or "notebooks" in project_root.parts[-1]:
            project_root = project_root.parent

    target_path = project_root / path

    if "models" in target_path.parts:
        if target_path.suffix not in ['.pt', '.pth'] and not target_path.is_dir():
            target_path = target_path.with_suffix('.pt')

    return target_path
