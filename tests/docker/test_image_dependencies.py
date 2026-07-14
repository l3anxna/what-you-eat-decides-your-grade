import importlib

import pytest

CORE_PACKAGES = ["numpy", "pandas", "torch"]
EDA_ONLY_PACKAGES = ["matplotlib", "seaborn"]


@pytest.mark.parametrize("package", CORE_PACKAGES)
def test_core_dependency_importable(package):
    importlib.import_module(package)


@pytest.mark.parametrize("package", EDA_ONLY_PACKAGES)
def test_eda_dependency_not_shipped(package):
    with pytest.raises(ImportError):
        importlib.import_module(package)
