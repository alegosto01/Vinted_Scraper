# Vinted New Version - Claude Instructions

This branch is optimized for low-token Claude Code work with Graphify.

## Start Here

Read `docs/AI_CONTEXT.md` for the compact project map. Do not read the full `wiki/`, `data/`, or `graphify-out/` directories unless the user explicitly asks.

## Graphify

This project has a local knowledge graph in `graphify-out/`.

For codebase questions, use Graphify before broad file reads:

```bash
graphify query "<question>" --budget 1500
graphify explain "<concept>"
graphify path "<A>" "<B>"
```

Read raw files only after Graphify narrows the relevant modules. Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.

After modifying code, refresh the local graph:

```bash
graphify update . --force
```

## Token Boundaries

Do not scan these by default:

- `data/`
- `wiki/`
- `graphify-out/`
- generated CSV, SQLite, model, report, image, and media artifacts

The wiki files are personal narrative notes and are not the default context source on this branch.

## Project

Vinted deal-finder: scraping, ML model training, live paper-trading, eventual-sale checks, and ranking experiments for fast-selling listings.

High-value entry points:

- `scripts/workflow_runner.py`
- `scripts/main.py`
- `scripts/simple_scraper.py`
- `scripts/analysis_pipeline/`
- `scripts/experiments/current/benchmark_basic_to_full/`
- `scripts/experiments/current/full_scrape_model/`
- `app.py`
- `tests/`

## Environment

- Python: `/home/ale/miniconda3/envs/vinted_scraper/bin/python`
- Main wrapper: `scripts/workflow_runner.py`
- Dashboard: `app.py`
- Searches config: `data/searches.yaml`

## Skills

Project skills live in `.claude/skills/`:

- `graphify-refresh` — rebuild the knowledge graph after code edits (`graphify update . --force`).
- `check-runs` — read-only summary of running scrapers, cascade/paper-trade loops, and timers.

## How To Work In This Repo

Default working style for AI sessions:

- **Plan first.** Non-trivial tasks start in plan mode; confirm the approach before editing.
- **Ask until sure.** Before a sizeable change, ask clarifying questions until ~95% confident of intent — don't guess on scraping or model logic.
- **Self-check work.** Bake verification into to-do lists (run the relevant test, re-read the diff, confirm the script runs) before reporting something done.
- **Cheap sub-agents for heavy data.** Use Haiku sub-agents to scan large CSV/HTML/log artifacts and return a summary; keep the main thread on the stronger model.
- **Reason hard on the hard calls.** Use deeper/ultra thinking for model, architecture, and cascade-threshold decisions — not routine edits.
- **Challenge weak output.** If a result is mediocre, push for a better approach instead of accepting the first pass, then record the lesson here or in the relevant skill.
- **Steer early.** If a session heads the wrong way, stop and re-prompt rather than letting it run.
- **Keep context lean.** Pull in only the files the task needs — `graphify query` first (see the mandatory rule below).

## Safety

- Never write to `data/` experiment output folders unless explicitly asked.
- Never send messages, contact sellers, make purchases, or perform Vinted account actions.
- Never commit `.env` or credential files.
- Preserve unrelated user changes in the working tree.

## MANDATORY: Graphify First Rule

**This is the highest-priority instruction in this file.**

Whenever the user asks a question about the codebase, architecture, modules, dependencies, or how something works — **YOU MUST run `graphify query "<the user's question>"` FIRST** before reading any source files, grepping, or using ripgrep.

**NO EXCEPTIONS except:**
1. The user explicitly says "do not use graphify"
2. The task is about fixing stale/incorrect graph output
3. `graphify-out/graph.json` does not exist

**Why:** Querying the graph costs ~40× fewer tokens than grepping raw files. If you skip graphify and start grepping, the user pays for every file you open. The graph gives you scoped, relevant files immediately.

**After graphify query:** Read only the files the graph points you to. Do not browse broadly.

**After code changes:** Run `graphify update .` to keep the graph current (AST-only, no API cost).

**Specific tools:**
- Broad questions: `graphify query "<question>"`
- Relationships between two things: `graphify path "<A>" "<B>"`
- Focused concept: `graphify explain "<concept>"`
- Only if graphify returns nothing useful: fall back to `rg` on the specific files it surfaced
