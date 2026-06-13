"""Compatibility wrapper for the moved experiment module."""
from pathlib import Path as _Path
import sys as _sys
for _parent in _Path(__file__).resolve().parents:
    if _parent.name == "scripts":
        if str(_parent) not in _sys.path:
            _sys.path.insert(0, str(_parent))
        break
from experiments._compat import export_module as _export_module, run_module_main as _run_module_main
_export_module('experiments.current.basic_5_giant_model.report_weighted_voting', globals())

if __name__ == "__main__":
    raise SystemExit(_run_module_main('experiments.current.basic_5_giant_model.report_weighted_voting'))
