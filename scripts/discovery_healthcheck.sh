#!/usr/bin/env bash
# 30-min health check for vinted-discovery-local.service.
# Gathers loop state, asks codex (gpt-5.4-codex, high reasoning) to diagnose and
# auto-fix if broken (may edit files + restart the service, MUST NOT git commit),
# then alerts via Telegram on any non-OK verdict. Runs from cron (no TTY).
set -uo pipefail

export HOME=/home/ale
export PATH=/home/ale/.npm-global/bin:/home/ale/miniconda3/envs/vinted_scraper/bin:/usr/bin:/bin
export XDG_RUNTIME_DIR=/run/user/1000

REPO=/home/ale/Desktop/vinted/Vinted_New_Version
UNIT=vinted-discovery-local.service
OUT="$REPO/experiments/current/giant_basic_visual/data/fast_discovery_local"
STATE_LOG=/home/ale/.cache/vinted-discovery-healthcheck.log

notify() {  # Slack alert -> GMEG #personal_workflow via incoming webhook
  local msg="$1"
  local hook
  hook=$(grep -E '^SLACK_WEBHOOK_URL=' "$REPO/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')
  [ -n "$hook" ] && curl -s --max-time 20 -X POST -H 'Content-type: application/json' \
    --data "$(python3 -c 'import json,sys;print(json.dumps({"text":sys.argv[1]}))' "$msg")" \
    "$hook" >/dev/null 2>&1
}

# --- gather state ---
ACTIVE=$(systemctl --user is-active "$UNIT" 2>&1)
JOURNAL=$(journalctl --user -u "$UNIT" --no-pager -n 40 2>&1 | tail -40)
REQ_CSV="$OUT/requests.csv"
if [ -f "$REQ_CSV" ]; then
  AGE_MIN=$(( ( $(date +%s) - $(stat -c %Y "$REQ_CSV") ) / 60 ))
  LAST_SCAN=$(tail -1 "$REQ_CSV")
else
  AGE_MIN=-1; LAST_SCAN="(no requests.csv yet)"
fi

# --- prompt ---
PROMPT=$(cat <<EOF
You are an unattended ops agent. Health-check the systemd user service '$UNIT'.

Expected healthy behavior:
- Service active (running).
- It runs experiments/current/giant_basic_visual/fast_discovery_loop.py --loop --interval 15
  --pages 3 --no-proxy --no-score for searches hobby_collezionismo, donna_accessori_gioielli,
  telefoni. It scrapes with NO proxy (curl_cffi).
- Every ~15 min it logs a line 'scan: {...} requests_this_scan=9'. requests.csv should update
  every ~15 min (so age well under ~20 min once warmed up). No 'Traceback', no repeated
  'SOFT_BLOCK'/'noproxy fetch err', no persistent GAP warnings.

Current observed state:
- systemctl is-active: $ACTIVE
- requests.csv age (min): $AGE_MIN   last row: $LAST_SCAN
- recent journal (last 40 lines):
$JOURNAL

Task: decide if it is healthy. If broken, FIX IT: you may edit any file in $REPO and run
'systemctl --user restart $UNIT'. Hard rules: do NOT run 'git commit', 'git add', 'git push',
or any git write. Keep changes minimal. If the fetch is soft-blocked, lowering --pages in the
unit is acceptable; run 'systemctl --user daemon-reload' after editing the unit.

End your reply with EXACTLY ONE final line, one of:
VERDICT: OK
VERDICT: FIXED - <what you changed>
VERDICT: BROKEN - <why it needs a human>
EOF
)

# --- run codex ---
# Model: gpt-5.6-sol is the ONLY model this ChatGPT account can use (codex-family,
# *-mini, *-nano, sol-mini all require API-key billing). Cheapest lever left is
# reasoning effort -> low. stdin closed so exec doesn't block.
RESP=$(cd "$REPO" && codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  -C "$REPO" \
  -m gpt-5.6-sol \
  -c model_reasoning_effort=low \
  "$PROMPT" < /dev/null 2>&1)
RC=$?

# Accept a verdict only from a successful codex run, and never match the prompt's
# own placeholder lines (those contain '<'). Empty verdict => treated as failure.
if [ "$RC" -eq 0 ]; then
  VERDICT=$(printf '%s\n' "$RESP" | grep -oE 'VERDICT: (OK|FIXED|BROKEN)[^<]*' | tail -1)
else
  VERDICT=""
fi
TS=$(date -Is)
printf '%s | rc=%s | %s\n' "$TS" "$RC" "${VERDICT:-<no verdict>}" >> "$STATE_LOG"

case "$VERDICT" in
  *"VERDICT: OK"*) : ;;                                   # healthy -> silent
  "" ) notify "⚠️ Vinted discovery healthcheck: codex returned no verdict (rc=$RC). Likely not logged in (run 'codex login') or model id wrong. active=$ACTIVE age=${AGE_MIN}m" ;;
  * )  notify "🔧 Vinted discovery healthcheck ($TS): $VERDICT" ;;
esac
