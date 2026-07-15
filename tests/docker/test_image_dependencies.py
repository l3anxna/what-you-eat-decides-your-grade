import importlib

import pytest

CORE_PACKAGES = ["numpy", "pandas", "torch", "joblib"]
ML_PACKAGES = ["sklearn", "xgboost", "shap"]
EDA_ONLY_PACKAGES = ["matplotlib", "seaborn"]


@pytest.mark.parametrize("package", CORE_PACKAGES)
def test_core_dependency_importable(package):
    importlib.import_module(package)


@pytest.mark.parametrize("package", ML_PACKAGES)
def test_ml_dependency_importable(package):
    """The image builds the pipeline (trains + saves all 5 models) and the CLI needs to
    load sklearn/xgboost models at runtime, so the `ml` group must be shipped, not just eda."""
    importlib.import_module(package)


@pytest.mark.parametrize("package", EDA_ONLY_PACKAGES)
def test_eda_dependency_not_shipped(package):
    with pytest.raises(ImportError):
        importlib.import_module(package)
