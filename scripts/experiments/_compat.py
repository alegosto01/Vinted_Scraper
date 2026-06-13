"""Compatibility helpers for moved experiment packages."""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_SKIP_EXPORTS = {"__name__", "__package__", "__loader__", "__spec__", "__builtins__"}


def export_module(target: str, namespace: dict[str, Any]) -> ModuleType:
    """Import ``target`` and expose its public runtime symbols in ``namespace``."""
    module = importlib.import_module(target)
    for name, value in vars(module).items():
        if name in _SKIP_EXPORTS:
            continue
        namespace[name] = value
    namespace["_COMPAT_TARGET_MODULE"] = module
    return module


def run_module_main(target: str) -> int | None:
    """Run target.main() for legacy direct-script wrappers."""
    module = importlib.import_module(target)
    main = getattr(module, "main", None)
    if main is None:
        raise SystemExit(f"Compatibility wrapper target {target!r} does not define main().")
    return main()
