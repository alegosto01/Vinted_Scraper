"""Compat shim: clustering_approach moved to experiments.old.clustering_approach."""
from experiments._compat import export_module as _export_module

_export_module("experiments.old.clustering_approach", globals())
