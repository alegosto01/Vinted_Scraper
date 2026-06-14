"""Compatibility wrapper for the moved (retired) experiment module."""
from pathlib import Path as _Path
import sys as _sys
for _parent in _Path(__file__).resolve().parents:
    if _parent.name == "scripts":
        for _candidate in (_parent.parent, _parent):
            if str(_candidate) not in _sys.path:
                _sys.path.insert(0, str(_candidate))
        break
from experiments._compat import export_module as _export_module, run_module_main as _run_module_main
_export_module('experiments.old.full_scrape_reranker.full_scrape_reranker', globals())

if __name__ == "__main__":
    raise SystemExit(_run_module_main('experiments.old.full_scrape_reranker.full_scrape_reranker'))
