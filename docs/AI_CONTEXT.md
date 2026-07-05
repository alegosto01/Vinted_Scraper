# AI Context

This branch is optimized for low-token Codex work with Graphify.

## Working Rule

Use Graphify for broad codebase questions before reading many files:

```bash
python scripts/dev/graphify_first.py "where is cascade scoring implemented?"
graphify query "where is cascade scoring implemented?"
graphify explain "cascade_runner"
graphify path "cascade_runner" "report_live_score_distributions"
```

Read raw files only after Graphify narrows the relevant modules.
See `docs/GRAPHIFY_WORKFLOW.md` for the practical low-token workflow.

## Assistant Integrations

- Codex: `AGENTS.md` and `.codex/hooks.json`
- Claude Code: `CLAUDE.md` and `.claude/settings.json`
- Kimi Code: `.kimi/skills/graphify/SKILL.md`
- Repo Codex skills: `.agents/skills/check-live-runs/`,
  `.agents/skills/giant-model-results/`, `.agents/skills/telegram-policy/`
- Read-only automation prompts: `docs/CODEX_AUTOMATIONS.md`

All three should use `graphify query`, `graphify explain`, or `graphify path` before broad source reads.

## Project Purpose

This project scrapes Vinted search snapshots, tracks listing status over time, ranks likely deals, evaluates whether ranked items later sell, and supports deeper final scraping for shortlisted buy candidates.

## High-Value Entry Points

| Area | Files |
|---|---|
| Long-running scraper | `scripts/main.py`, `scripts/simple_scraper.py` |
| Pipeline wrapper | `scripts/workflow_runner.py` |
| Deal scoring pipeline | `scripts/analysis_pipeline/` |
| Eventual-sale checks | `scripts/daily_eventual_sales.py`, `scripts/analysis_pipeline/evaluation/update_eventual_sales.py` |
| Benchmark cascade | `experiments/old/benchmark_basic_to_full/` |
| Full scrape model | `experiments/old/full_scrape_model/` |
| Streamlit UI | `app.py` |
| Tests | `tests/` |

## Runtime Artifacts

Stable scrape datasets live under `data/simple_scrape/`. Local process output lives under `runtime/`:

- `runtime/logs/simple_scrape/` for scraper telemetry, eventual-sale logs, and JSONL counters.
- `runtime/pids/` for process ID files.
- `runtime/archive/` for local rotated runtime history.

## Token-Saving Boundaries

Do not scan these unless the user asks for them explicitly:

- `data/`
- `runtime/`
- `graphify-out/`
- generated CSV, SQLite, image, model, and report artifacts

For project rules and common commands, prefer `docs/CODEX_WORKFLOW.md`.
