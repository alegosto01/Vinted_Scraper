#!/usr/bin/env bash
# Push collector run-dir CSV state laptop -> VPS (one-way).
# Run from laptop. Driven by cron (see README_DEPLOY.md).
set -euo pipefail

# --- edit these for your VPS ---
VPS_HOST="${VPS_HOST:-vinted@YOUR_VPS_IP}"
VPS_REPO="${VPS_REPO:-/home/vinted/Vinted_New_Version}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
# --------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_REL="data/experiments/time_to_sell/live_runs/bin_collector_20260602_214104"
EXCLUDE="$REPO_ROOT/deploy/rsync-exclude.txt"

# --mkpath needs rsync >= 3.2.3; if older, create the dir on VPS once by hand.
rsync -az --delete-after \
  --exclude-from="$EXCLUDE" \
  -e "ssh -i $SSH_KEY -o BatchMode=yes" \
  "$REPO_ROOT/$RUN_REL/" \
  "$VPS_HOST:$VPS_REPO/$RUN_REL/"

echo "[sync] $(date -Is) pushed $RUN_REL to $VPS_HOST"
