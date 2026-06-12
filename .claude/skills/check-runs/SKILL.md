---
name: check-runs
description: Summarize the deal-finder's running background jobs — scrapers, cascade live runners, paper-trade loops, the Streamlit dashboard, and scheduled timers. Use to see what is currently active before starting or stopping anything.
---

# Check background runs

This project runs many long-lived jobs (the long-running scraper, cascade live runners,
paper-trade loops, the Streamlit app, scheduled timers). This skill reports what is currently
active. Everything here is **read-only** — never kill, restart, or modify a job unless the user
explicitly asks.

## Steps

Run these and summarize the results together:

1. tmux sessions:
   ```bash
   tmux ls 2>/dev/null || echo "no tmux server"
   ```

2. Project python processes (with elapsed time):
   ```bash
   ps -eo pid,etime,cmd --no-headers | grep -E "scripts/(main|simple_scraper|workflow_runner|daily_eventual_sales)|cascade_runner|paper_trade|app\.py" | grep -v grep || echo "no matching python processes"
   ```

3. User scheduled timers:
   ```bash
   systemctl --user list-timers --no-pager 2>/dev/null | head -20 || echo "no user timers"
   ```

## Report

- Which jobs are up and how long each has been running (`etime`).
- Flag anything suspicious: a job that looks **stuck** (very long runtime, no expected churn),
  or the **same runner started twice** (duplicate scrape/paper-trade processes waste rate limit
  and can corrupt shared output).
- Do not take action — just summarize and, if something looks wrong, recommend what the user
  could do.
