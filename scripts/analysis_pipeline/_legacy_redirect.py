from __future__ import annotations

import runpy
from importlib import import_module


def load_legacy_module(namespace: dict, target_module: str) -> None:
    if namespace.get("__name__") == "__main__":
        return
    module = import_module(target_module)
    namespace["__doc__"] = getattr(module, "__doc__", namespace.get("__doc__"))
    if hasattr(module, "__all__"):
        namespace["__all__"] = getattr(module, "__all__")
    for key, value in module.__dict__.items():
        if key.startswith("__"):
            continue
        namespace[key] = value


def run_legacy_module(target_module: str) -> None:
    runpy.run_module(target_module, run_name="__main__")
