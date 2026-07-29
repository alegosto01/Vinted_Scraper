"""log_watcher.py — Real-time log monitor.

Tails all process logs, classifies suspicious lines with Gemini, then asks
Claude to diagnose confirmed bugs and sends a Telegram alert.

Run:
    nohup /home/ale/miniconda3/envs/vinted_scraper/bin/python \
        scripts/telegram_implementation/log_watcher.py \
        >> data/telegram_implementation/watcher.log 2>&1 &

Maintenance channel: set MAINTENANCE_CHAT_ID env var to a different chat/group
id; defaults to the same chat_id as regular bot notifications.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import httpx

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("log_watcher")

# ── Runtime config ─────────────────────────────────────────────────────────────
_BOT_TOKEN: str = settings.telegram.bot_token or ""
_MAIN_CHAT_ID: str = str(settings.telegram.chat_id or os.getenv("CHAT_ID", ""))
_MAINT_CHAT_ID: str = os.getenv("MAINTENANCE_CHAT_ID", _MAIN_CHAT_ID)
_GEMINI_KEY: str = (os.getenv("GEMINI_API_KEY") or "").strip()
_GEMINI_MODEL = "gemini-2.5-flash"
_CLAUDE_CLI = Path(
    os.getenv(
        "CLAUDE_CLI",
        "/home/ale/.vscode/extensions/anthropic.claude-code-2.1.143-linux-x64/resources/native-binary/claude",
    )
)
_DEDUP_WINDOW = 1800  # seconds before re-alerting on the same error signature

# ── Watched logs ───────────────────────────────────────────────────────────────
_WATCHED: dict[str, Path] = {
    "cascade": PROJECT_ROOT / "experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live/cascade_loop.log",
    "bot": PROJECT_ROOT / "data/telegram_implementation/bot.log",
    "scraper": settings.paths.scraper_log_path,
    "eventual_sales": settings.paths.eventual_sales_log_path,
}

# ── Known-OK noise: skip Gemini entirely ──────────────────────────────────────
_NOISE = re.compile(
    # External-service transient failures (proxy / DNS / network) — self-resolving
    r"502 Zone has reached usage limit"
    r"|Skipping empty catalog response"
    r"|Datacenter fetch (?:error|failed)"
    r"|HTTPSConnectionPool.*Max retries"
    r"|Caused by ProxyError"
    r"|Tunnel connection failed"
    r"|Temporary failure in name resolution"
    r"|NewConnectionError"
    r"|Failed to establish a new connection"
    # Standard log levels and HTTP traffic
    r"|\bINFO\b"
    r"|\bDEBUG\b"
    r"|HTTP Request.*(?:200 OK|getUpdates)"
    r"|Application (?:start|stop)"
    r"|Bot starting"
    r"|getUpdates"
    # Python / pandas warnings (multi-line blocks)
    r"|\bDeprecationWarning\b"
    r"|\bUserWarning\b"
    r"|\bDtypeWarning\b"
    r"|\bFutureWarning\b"
    r"|\bRuntimeWarning\b"
    r"|\bPerformanceWarning\b"
    r"|\bSettingWithCopyWarning\b"
    r"|\.py:\d+:\s+\w*Warning"            # header line of a warning block, e.g. `foo.py:123: UserWarning: ...`
    r"|^\s+\w+\s*=\s*pd\."                  # continuation line of a warning block, e.g. `  last_old_date = pd.to_datetime(...)`
    # Routine scraper / scheduler info lines (no level prefix)
    r"|Completed scheduler iteration"
    r"|Scheduler (?:summary|iteration)"
    r"|Updated schedule for search="
    r"|No deal-finder model configured"
    r"|Deal-finder scoring search="
    r"|Finished (?:initial scrape|post-processing)"
    r"|Appending \d+ new items"
    r"|No searches due right now"
    r"|Search=.* time difference"
    # Routine eventual-sale background traffic
    r"|Background eventual-sale (?:check|result)"
    r"|Checking \d+ items? if they are sold"
    r"|Checking if item .* is sold"
    r"|Checked item .*: status="
    # Expected Italian listing text the prior prompt called out
    r"|Nessuna recensione",
    re.IGNORECASE | re.MULTILINE,
)

# ── Per-log context buffer and dedup state ────────────────────────────────────
_ctx: dict[str, deque[str]] = {n: deque(maxlen=50) for n in _WATCHED}
_seen: dict[str, float] = {}


def _is_noise(line: str) -> bool:
    return len(line.strip()) < 8 or bool(_NOISE.search(line))


def _sig(log_name: str, line: str) -> str:
    """Stable hash that ignores timestamps, PIDs, and memory addresses."""
    normalised = re.sub(
        r"0x[0-9a-fA-F]+"
        r"|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[,.\d]*"
        r"|pid=\d+|PID\s+\d+"
        r"|\b\d{5,}\b",
        "X",
        line,
    )
    return hashlib.md5(f"{log_name}:{normalised}".encode()).hexdigest()[:12]


def _is_dup(sig: str) -> bool:
    now = time.monotonic()
    last = _seen.get(sig)
    if last and (now - last) < _DEDUP_WINDOW:
        return True
    _seen[sig] = now
    return False


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Gemini: is this line a real bug? ──────────────────────────────────────────
async def _classify(log_name: str, line: str, ctx: list[str]) -> tuple[bool, str, str]:
    """Returns (is_bug, severity, reason)."""
    context_text = "".join(ctx[-25:])
    prompt = (
        "You monitor a Vinted reselling automation system running on Linux.\n"
        "A noise regex has already filtered out routine INFO/DEBUG logs, pandas warnings, "
        "scheduler status lines, eventual-sale background checks, and known transient external "
        "errors (proxy 502, DNS failures). Any line you see has already passed that first filter, "
        "so lean toward classifying as a bug when the evidence is plausible.\n\n"
        f"Log source: [{log_name}]\n\n"
        f"Recent context:\n{context_text}\n"
        f"Suspicious new line:\n{line}\n\n"
        "Classify as a BUG (is_bug=true) when the line shows any of:\n"
        "- A Python traceback (`Traceback (most recent call last)`) or any exception name "
        "(KeyError, ValueError, AttributeError, TypeError, RuntimeError, ConnectionError, etc.)\n"
        "- An ERROR-level log line that references a crash, aborted processing, or unhandled state\n"
        "- A data integrity issue (missing column, schema mismatch, wrong row count, NaN where required)\n"
        "- A scraper or pipeline step that explicitly reports failure beyond a single retry\n\n"
        "Classify as NOT a bug (is_bug=false) only when the line is clearly:\n"
        "- A transient external API or network blip the regex didn't catch\n"
        "- An informational status line that snuck through the noise filter\n"
        "- A test or simulation marker the operator deliberately injected\n\n"
        'Reply ONLY with JSON: {"is_bug": bool, "severity": "low|medium|high", "reason": "one sentence"}'
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_GEMINI_MODEL}:generateContent?key={_GEMINI_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 96},
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            if m:
                d = json.loads(m.group())
                return bool(d.get("is_bug")), str(d.get("severity", "medium")), str(d.get("reason", ""))
    except Exception as exc:
        LOGGER.warning("Gemini classify error: %s", exc)
    return False, "", ""


# ── Claude: diagnose and suggest a fix ────────────────────────────────────────
async def _diagnose(log_name: str, line: str, ctx: list[str], reason: str) -> str:
    if not _CLAUDE_CLI.exists():
        LOGGER.warning("Claude CLI not found at %s", _CLAUDE_CLI)
        return ""
    context_text = "".join(ctx[-30:])
    prompt = (
        f"A log monitor detected a bug in the Vinted automation system.\n"
        f"Working directory: {PROJECT_ROOT}\n"
        f"Log source: [{log_name}]\n"
        f"Classifier reason: {reason}\n\n"
        f"Error line:\n{line}\n\n"
        f"Log context (last 30 lines):\n{context_text}\n\n"
        "Please:\n"
        "1. Identify which file and line number is causing this\n"
        "2. Explain what's wrong\n"
        "3. Show the exact code fix\n\n"
        "Be concise — this goes into a Telegram message. Max 350 words."
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            str(_CLAUDE_CLI), "-p",
            "--dangerously-skip-permissions",
            "--allowedTools", "Read,Bash(cat *),Bash(grep *),Bash(ls *)",
            "--add-dir", str(PROJECT_ROOT),
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        return stdout.decode().strip()[:2000]
    except asyncio.TimeoutError:
        return "(Claude diagnosis timed out after 120s)"
    except Exception as exc:
        LOGGER.warning("Claude diagnose error: %s", exc)
        return ""


# ── Telegram alert ─────────────────────────────────────────────────────────────
def _redact(value: object) -> str:
    """httpx exceptions can carry the request URL, and the URL carries the token."""
    text = str(value)
    return text.replace(_BOT_TOKEN, "***") if _BOT_TOKEN else text


async def _alert(text: str) -> None:
    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    for start in range(0, len(text), 4000):
        chunk = text[start : start + 4000]
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                await client.post(url, json={"chat_id": _MAINT_CHAT_ID, "text": chunk, "parse_mode": "HTML"})
            except Exception as exc:
                LOGGER.warning("Telegram alert failed: %s", _redact(exc))


# ── Per-line pipeline ──────────────────────────────────────────────────────────
async def _handle(log_name: str, line: str) -> None:
    ctx = list(_ctx[log_name])

    is_bug, severity, reason = await _classify(log_name, line, ctx)
    if not is_bug:
        return

    sig = _sig(log_name, line)
    if _is_dup(sig):
        return

    LOGGER.warning("[%s] Bug confirmed (%s): %s", log_name, severity, line[:120])

    emoji = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(severity, "🟡")
    parts = [
        f"{emoji} <b>Bug in [{log_name}]</b> — {severity}",
        f"<code>{_esc(line[:400])}</code>",
        f"<i>{_esc(reason)}</i>",
    ]

    diagnosis = await _diagnose(log_name, line, ctx, reason)
    if diagnosis:
        parts.append(f"\n🤖 <b>Claude diagnosis:</b>\n<pre>{_esc(diagnosis[:1500])}</pre>")

    await _alert("\n".join(parts))


# ── File tailer ────────────────────────────────────────────────────────────────
async def _tail(log_name: str, log_path: Path) -> None:
    LOGGER.info("Watching [%s] → %s", log_name, log_path)
    while True:
        if not log_path.exists():
            await asyncio.sleep(10)
            continue
        try:
            with log_path.open("r", errors="replace") as fh:
                fh.seek(0, 2)  # start at EOF — only watch new lines
                last_size = fh.tell()
                while True:
                    line = fh.readline()
                    if not line:
                        try:
                            cur_size = log_path.stat().st_size
                        except FileNotFoundError:
                            break
                        if cur_size < last_size:  # log rotated / truncated
                            fh.seek(0)
                        last_size = fh.tell()
                        await asyncio.sleep(0.5)
                        continue
                    line = line.rstrip("\n")
                    _ctx[log_name].append(line + "\n")
                    if not _is_noise(line):
                        asyncio.create_task(_handle(log_name, line))
        except Exception as exc:
            LOGGER.error("[%s] tailer crashed: %s — restarting in 5s", log_name, exc)
            await asyncio.sleep(5)


# ── Entry point ────────────────────────────────────────────────────────────────
async def _main() -> None:
    active = {n: p for n, p in _WATCHED.items() if p.exists()}
    missing = set(_WATCHED) - set(active)
    LOGGER.info("Starting — %d active logs, %d not yet found", len(active), len(missing))

    if not _BOT_TOKEN:
        LOGGER.error("BOT_TOKEN not configured — cannot send Telegram alerts")

    await _alert(
        "🟢 <b>Log watcher started</b>\n"
        + f"Active: {', '.join(active) or 'none'}\n"
        + (f"Pending (will watch once created): {', '.join(missing)}" if missing else "")
    )

    await asyncio.gather(*[asyncio.create_task(_tail(n, p)) for n, p in _WATCHED.items()])


if __name__ == "__main__":
    asyncio.run(_main())
