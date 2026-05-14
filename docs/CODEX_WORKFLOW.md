# Codex Workflow

This file explains how Codex should work in this project so future changes stay consistent, documented, and easy to resume.

## Main Goal

Help the project move forward without losing context.

When changing code, data rules, commands, or pipeline behavior, also update the relevant Markdown docs.

## Documentation Rule

If behavior changes, update the docs in the same work session.

Common doc targets:

| If This Changes | Update This |
|---|---|
| Pipeline command or order | `docs/PIPELINE.md` |
| Meaning/location of CSVs | `docs/DATA_FILES.md` |
| Eventual-sale behavior | `docs/EVENTUAL_SALES.md` |
| Common error or debugging process | `docs/TROUBLESHOOTING.md` |
| Remote setup or SSH/Tailscale flow | `docs/REMOTE_ACCESS.md` |
| Project-level explanation | `docs/PROJECT_OVERVIEW.md` |
| New future idea | `docs/IDEAS.md` |
| Deal-finder experiment behavior | `docs/EXPERIMENTS.md` |
| Paper-trading behavior | `docs/LIVE_DEAL_TESTING.md` |
| Photo-improvement experiment behavior | `docs/PHOTO_ARBITRAGE.md` |
| Prompt wording or communication safety | `docs/COMMUNICATION_SAFETY.md` |
| Long-chat handoff or active run context | `docs/SESSION_CONTEXT.md` |

## Project Rules To Preserve

- Rows already in `sold_df.csv` should not also appear in `eventual_sale_check/sold_eventually.csv`.
- Use `Dataid` and `Link` as the main item identity columns when checking overlap or duplicates.
- Duplicate-looking rows are not necessarily true duplicates if `Dataid` or `Link` differs.
- Full seller/item scraping usually happens during `final-buy-filter`.
- Prefer using `scripts/workflow_runner.py` for pipeline commands when possible.
- Long-running scraper work should be run in a persistent session such as `tmux`.
- Deal-finder experiments must write under `data/experiments/deal_finder/`.
- Paper-trading must not update production tracking CSVs such as `old_df.csv`, `sold_df.csv`, `unsold_df.csv`, or `big_raw.csv`.
- Deal-finder offline training uses saved sold/not-sold labels; the 2-day success target is evaluated during paper-trading, not assumed from historical data.
- Prompts and summaries should frame this as local data science / paper-trading on public listing snapshots; avoid wording that sounds like security testing, evasion, credentials, account automation, or abuse.
- For larger experiment requests, reuse the wrapper and templates in `docs/COMMUNICATION_SAFETY.md` instead of pasting older prompts that mention network setup, accounts, or provider details.
- When a conversation becomes long or before switching chats, update `docs/SESSION_CONTEXT.md` and run `python3 scripts/save_context_snapshot.py`.

## Preferred Work Pattern

1. Inspect the relevant code/data before making assumptions.
2. Make the smallest safe change that fixes the issue.
3. Preserve user data and avoid reverting unrelated changes.
4. Add or update tests when behavior changes.
5. Update Markdown docs if the project knowledge changed.
6. Summarize clearly what changed and what was verified.

## Useful Verification Commands

Refresh the long-chat handoff snapshot:

```bash
python3 scripts/save_context_snapshot.py
```

Run scraper/eventual-sale tests:

```bash
python3 -m unittest tests.test_daily_eventual_sales tests.test_update_eventual_sales tests.test_sold_csv_recheck tests.test_simple_scraper
```

Run compile checks:

```bash
python3 -m py_compile scripts/simple_scraper.py scripts/scraping_options.py scripts/daily_eventual_sales.py scripts/workflow_runner.py
```

Check active scraper/background processes:

```bash
ps -ef | rg 'scripts/main.py|workflow_runner.py main|daily_eventual_sales|python'
```

## Final Answer Checklist

When Codex finishes a task, mention:

- What changed.
- Which important files were changed.
- What tests or checks were run.
- Any remaining risks or things to verify.

Keep the final answer short unless the user asks for detail.

## How To Use `docs/IDEAS.md`

Use `docs/IDEAS.md` as the messy inbox for future work.

When an idea becomes a real rule, process, or decision, move it from `IDEAS.md` into the appropriate structured doc.

## Safe Prompt Wrapper

For long requests, start with:

```text
This is my own local Vinted_New_Version data-analysis project.
Work only with local project files and public listing snapshots.
Use the existing project configuration.
Do not edit private configuration files, credentials, or account settings.
Do not make purchases, contact sellers, send messages, or perform account actions.
Save experiment outputs only under data/experiments/deal_finder/ unless I explicitly approve another location.
```

Then add the specific task in simple data-analysis terms, such as checking benchmark results, adding an offline experiment, or updating the scheduled paper-trading job.
