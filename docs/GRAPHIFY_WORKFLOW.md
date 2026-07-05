# Graphify Workflow

Use Graphify as the first step for codebase questions. The goal is to avoid
reading broad folders and spend tokens only on the files that matter.

## Quick Start

```bash
python scripts/dev/graphify_first.py "where is Telegram sending decided?"
```

This runs:

```bash
graphify query "<question>" --budget 1500
```

Then read only the files or nodes Graphify surfaces.

## When To Use Each Command

| Need | Command |
|---|---|
| Find relevant files for a broad question | `graphify query "<question>" --budget 1500` |
| Understand one module/concept | `graphify explain "<node>"` |
| Understand how two components connect | `graphify path "<A>" "<B>"` |
| Refresh after code edits | `graphify update . --force` |

## Recommended Prompt

When asking an assistant about this repo, start with:

> Use Graphify first and avoid broad file reads. Question: ...

The assistant should run one Graphify query, read only the files it identifies,
and fall back to `rg` only if the graph is stale or not specific enough.

## What Not To Do

- Do not start by reading `graphify-out/GRAPH_REPORT.md`; it is large.
- Do not ask the assistant to scan `data/`, `runtime/`, or `graphify-out/` unless
  that is exactly the task.
- Do not use Graphify for current live process status; use process/log commands
  for that.

## Wiki Relationship

Graphify is for finding code. The wiki notes are plain Markdown files for
preserving decisions and experiment summaries after the code question has been
answered.
