# Communication Safety

This project is local marketplace data science work: saved CSV analysis, offline model evaluation, and paper-trading on public listing snapshots.

When asking Codex to work on this project, keep prompts framed around local experiments and public data. Avoid wording that can be misread as security testing, account automation, evasion, credential work, or abuse.

## Default Prompt Wrapper

Start larger requests with this wrapper:

```text
This is my own local Vinted_New_Version data-analysis project.
Work only with local project files and public listing snapshots.
Use the existing project configuration.
Do not edit private configuration files, credentials, or account settings.
Do not make purchases, contact sellers, send messages, or perform account actions.
Save experiment outputs only under data/experiments/deal_finder/ unless I explicitly approve another location.
```

Then add the specific task, for example:

```text
Please check the current paper-trading benchmark results and summarize performance by search, model, and threshold.
```

## Preferred Framing

Use wording like:

- "local Vinted_New_Version project"
- "public first-page listing snapshots"
- "paper-trading benchmark"
- "offline model validation"
- "status recheck for tracked listings"
- "existing project configuration"
- "write only under `data/experiments/deal_finder/`"
- "do not contact sellers or perform account actions"

Avoid wording that suggests the task is about access, accounts, evasion, or aggressive collection. Prefer these safer replacements:

| Instead Of | Prefer |
|---|---|
| "live automation" | "scheduled local paper-trading job" |
| "scraping aggressively" | "collecting public listing snapshots at the approved cadence" |
| "proxy changes" | "existing project collection configuration" |
| "avoid blocks" | "keep the collector reliable and respectful of the configured cadence" |
| "account/login actions" | "no account actions" |
| "stealth/evasion language" | "normal local data collection within project rules" |
| "exploit/bypass language" | "debug the local collector or data pipeline" |

## Good Prompt Pattern

```text
Please continue the local deal-finder experiment.

Use only the existing project configuration.
Collect public first-page listing snapshots for paper-trading evaluation.
Save outputs only under data/experiments/deal_finder/.
Do not edit private configuration files.
Do not make purchases, contact sellers, or perform account actions.
Summarize the offline/live metrics and any files changed.
```

## Project Boundaries

Codex should preserve these boundaries:

- Do not edit `.env`, credentials, or private configuration unless explicitly approved.
- Do not add new paid providers or external services unless explicitly approved.
- Do not contact sellers, send messages, make purchases, or perform account actions.
- Do not delete production data.
- Do not write paper-trading outputs into production CSVs.
- Treat live work as observation and ranking only.

## If A Prompt Gets Flagged

Rewrite it to emphasize:

- It is a local project.
- It uses public listing snapshots.
- It is for ranking/model evaluation.
- It avoids account actions.
- It writes only to experiment folders.

Remove language that sounds like bypassing, evasion, unauthorized access, credentials, or aggressive automation.

## Safer Task Templates

Use these templates when possible.

### Check Current Results

```text
Please check the current local paper-trading benchmark results.
Summarize performance by search, model, and threshold.
Use only files under data/experiments/deal_finder/.
Do not change any files.
```

### Add Offline Experiment

```text
Please add an offline model-comparison experiment for my local saved CSV data.
Use only historical files already present in data/simple_scrape/.
Avoid future-label leakage.
Write outputs under data/experiments/deal_finder/.
Do not modify production CSVs or private configuration.
```

### Adjust Scheduled Paper-Trading

```text
Please update the scheduled local paper-trading benchmark.
Use public listing snapshots and the existing project configuration.
Save all benchmark outputs under data/experiments/deal_finder/.
Do not contact sellers, make purchases, send messages, or perform account actions.
```

### Normal Project Collection

```text
Please resume the usual local project collection process.
Use the existing project configuration.
Keep experiment outputs separate from production CSVs.
Do not edit private configuration files or credentials.
```

## Codex Response Style For This Project

When Codex summarizes work, prefer wording like:

- "local collector"
- "scheduled paper-trading job"
- "public listing snapshot"
- "status recheck"
- "model benchmark"
- "experiment output"

Avoid adding unnecessary details about network setup, account behavior, provider internals, or anything unrelated to local data analysis.
