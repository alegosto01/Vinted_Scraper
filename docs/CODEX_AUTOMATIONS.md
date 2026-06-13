# Codex Automations

Use automations only for read-only monitoring until the checks are boringly
reliable. Do not let scheduled automations restart processes, change
thresholds, send Telegram messages, edit credentials, or write production data.

## Recommended Automations

### Live Health Check

Cadence: every 6 hours.

```text
Use $check-live-runs.
Check whether the local collector, giant Basic5 scorer, and Telegram loop appear
to be running. Read only process lists and small log tails. Report only if a
process is missing, a log is stale, or a recent error appears. Do not restart,
kill, edit, or send anything.
```

### Daily Giant Model Summary

Cadence: every morning.

```text
Use $giant-model-results.
Summarize the latest Basic5 giant-model live results: evaluated rows, Telegram
candidates, sent count if available, best model overall, best model by search,
and any changed precision/recall pattern. Read existing outputs only. Do not
retrain, edit files, or send Telegram messages.
```

### Daily Sold Links Summary

Cadence: every morning after the live result summary.

```text
Use $giant-model-results.
List newly sold items from the latest matured live scoring outputs, including
search name, title, price, model/pass reason, and link when present. Read
existing outputs only. Do not recheck URLs or write files.
```

### Weekly Experiment Hygiene

Cadence: weekly.

```text
Use Graphify first.
Review current vs old experiment folders and docs. Report stale docs, wrappers,
or experiment folders that appear inconsistent with the current/old split. Do
not move, delete, or edit files.
```

## Safety Defaults

- Prefer standalone automation runs for summaries.
- Prefer thread automation only when monitoring a specific long-running task
  in an active conversation.
- Keep sandbox at workspace-write or read-only.
- Review the first few automation outputs before increasing cadence.
- If an automation needs to write files, convert it into a normal interactive
  Codex task first.
