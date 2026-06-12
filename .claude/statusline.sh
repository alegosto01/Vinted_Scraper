#!/usr/bin/env bash
# Claude Code status line — mini dashboard for the session.
# Receives a JSON payload on stdin (model, workspace, cost, context flags).
# Uses python3 (jq is not installed on this machine).
CC_STATUS_INPUT="$(cat)" python3 -c '
import os, json, subprocess

try:
    d = json.loads(os.environ.get("CC_STATUS_INPUT") or "{}")
except Exception:
    d = {}

model = (d.get("model") or {}).get("display_name") or (d.get("model") or {}).get("id") or "?"
cdir  = (d.get("workspace") or {}).get("current_dir") or d.get("cwd") or ""
cost  = (d.get("cost") or {}).get("total_cost_usd")
exceeds = d.get("exceeds_200k_tokens")

branch = ""
if cdir and os.path.isdir(cdir):
    try:
        branch = subprocess.check_output(
            ["git", "-C", cdir, "branch", "--show-current"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        branch = ""

parts = [f"\U0001f916 {model}"]
if cdir:
    parts.append(f"\U0001f4c1 {os.path.basename(cdir)}")
if branch:
    parts.append(f"⎎ {branch}")
if isinstance(cost, (int, float)):
    parts.append(f"\U0001f4b0 ${cost:.2f}")
if exceeds:
    parts.append("⚠️ ctx>200k")

print("  ".join(parts), end="")
'
