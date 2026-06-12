---
name: graphify-refresh
description: Rebuild the local Graphify knowledge graph after code changes so `graphify query`/`explain`/`path` stay accurate. Use whenever source files under scripts/, app.py, or tests/ have been edited in this session.
---

# Refresh the Graphify knowledge graph

This repo keeps a local knowledge graph in `graphify-out/`. After editing code, the graph
goes stale, which makes future `graphify query` / `explain` / `path` answers wrong. This skill
rebuilds it. The rebuild is AST-only — **no LLM/API cost.**

## Steps

1. Rebuild the graph (overwrite even if the new graph has fewer nodes, e.g. after deleting code):

   ```bash
   graphify update . --force
   ```

2. If the command errors, report the error verbatim — do **not** silently fall back to a full
   re-index or read `graphify-out/GRAPH_REPORT.md` (it is large and token-heavy).

3. Briefly confirm success and name the modules that changed, so the user knows the graph now
   reflects this session's edits.

## Notes

- Run this at the end of any task that modified `.py` files — it is the "After code changes"
  step from `CLAUDE.md`.
- For continuous rebuilding during a long editing session, `graphify watch .` can be left
  running instead, but the one-shot `update` above is the default.
