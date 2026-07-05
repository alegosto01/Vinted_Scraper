---
name: check-live-runs
description: Check this repo's local live collector, giant Basic5 scorer, Telegram loop, resume waiter, and related logs/PIDs. Use for "is it running?", "restart/check processes", "were they running?", collector log freshness, and read-only live health checks.
---

# Check Live Runs

Use this skill for read-only status checks of the local Vinted live processes.

## Rules

- Start with `graphify query` only if code structure is unclear. For plain process/log status, use process and log commands directly.
- Do not edit files, restart processes, kill processes, or send Telegram messages unless the user explicitly asks.
- Report exact timestamps and paths. Use Europe/Rome local time when interpreting "today", "yesterday", or "last N days".
- Treat these as the usual live surfaces:
  - collector: `experiments/current/time_to_sell/live_bin_collector.py`
  - giant scorer / Telegram loop: `experiments/current/basic_5_giant_model/apply_to_live_collector.py`
  - collector runs: `experiments/current/time_to_sell/data/live_runs/`
  - giant scoring: `experiments/current/basic_5_giant_model/data/live_scoring/`

## Workflow

1. List matching Python processes:

   ```bash
   ps -ef | rg 'live_bin_collector|apply_to_live_collector|telegram|basic_5_giant|time_to_sell|workflow_runner.py'
   ```

2. Find the latest relevant logs without opening broad folders:

   ```bash
   find experiments/current/time_to_sell/data/live_runs -maxdepth 3 -type f -name 'collector.log' -printf '%T@ %p\n' | sort -nr | head
   find experiments/current/basic_5_giant_model/data/live_scoring -maxdepth 2 -type f -name '*.log' -printf '%T@ %p\n' | sort -nr | head
   ```

3. Read only the latest small tails:

   ```bash
   tail -80 <latest-collector-log>
   tail -80 <latest-telegram-or-scorer-log>
   ```

4. Summarize:
   - process name/PID/command if running
   - latest log path and last modified time
   - last visible successful action
   - last visible error, if any
   - whether the evidence proves "running now" or only "ran recently"

## Finish With Verification

End with this read-only check unless a more specific command was required:

```bash
ps -ef | rg 'live_bin_collector|apply_to_live_collector|telegram|basic_5_giant|time_to_sell|workflow_runner.py'
```
