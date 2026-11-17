import importlib
from pathlib import Path


def pytest_sessionstart(session):
    root = Path(__file__).resolve().parents[2]
    packages = {
        "dodo_memory": root / "dodo_memory/__init__.py",
        "dodo_memory_desktop": root / "desktop/src/dodo_memory_desktop/__init__.py",
        "dodo_mem_neural": root / "neural/src/dodo_mem_neural/__init__.py",
    }
    for name, expected in packages.items():
        assert Path(importlib.import_module(name).__file__).resolve() == expected
