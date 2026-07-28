"""Slow local bad-photo deal monitor for Vinted.

One Vinted catalog request is made at most once per minute. New listings with
MUSIQ below the configured threshold get a condition-locked title search,
E5+DINOv2 comparison, mobile HTML report, and optional Telegram alert.
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import ipaddress
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests_html
from curl_cffi import requests as cffi
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scripts"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))

from config.project_config import settings  # noqa: E402
from config.search_loader import load_searches  # noqa: E402
from match_vinted_products import (  # noqa: E402
    DINOImageEncoder,
    E5TitleEncoder,
    MatchConfig,
    choose_device,
    run_matching,
)
from full_compare import FullSplitConfig, score_full, write_dual_report  # noqa: E402
from full_item_fetch import fetch_item  # noqa: E402
from spec_compare import compare_specs, usable_comparables, verdict_summary  # noqa: E402
from title_query import rewrite_title  # noqa: E402
from scrape_title_candidates import (  # noqa: E402
    HEADERS,
    PRODUCT_SELECTOR,
    STATUS_NAMES,
    catalog_url,
    parse_condition,
)
from simple_scraper import Simple_scraper  # noqa: E402

LOG = logging.getLogger("bad_photo_deal_monitor")
SEARCHES = (
    "telefoni",
    "griffati_uomo_all",
    "griffati_donna_all",
    "gucci",
    "prada",
    "nike",
    "ps4",
    "donna_accessori_gioielli",
)
CONDITION_IDS = {name.casefold(): status_id for status_id, name in STATUS_NAMES.items()}
MAX_FULL_COMPARE_ATTEMPTS = 5
MUSIQ_BAD_THRESHOLD = 55.0
# Phone photos score lower across the board, so they need their own cut.
SEARCH_MUSIQ_THRESHOLDS = {"telefoni": 65.0}


@dataclass(frozen=True)
class SplitConfig:
    title_min: float = 0.82
    image_min: float = 0.55
    combined_min: float = 0.70
    min_kept_for_verdict: int = 3


class RequestPacer:
    def __init__(self, gap_seconds: float, label: str = "catalog"):
        self.gap_seconds = gap_seconds
        self.label = label
        self.last_request_at: float | None = None

    def wait(self) -> None:
        if self.last_request_at is not None:
            remaining = self.gap_seconds - (time.monotonic() - self.last_request_at)
            if remaining > 0:
                LOG.info("Chill pace: waiting %.1fs before next %s request", remaining, self.label)
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


class CatalogClient:
    def __init__(self, gap_seconds: float, block_pause_seconds: float = 600.0,
                 fallback_seconds: float = 600.0):
        self.session = cffi.Session(impersonate="chrome")
        self.pacer = RequestPacer(gap_seconds)
        self.scraper = Simple_scraper()
        self.block_pause_seconds = block_pause_seconds
        self.fallback_seconds = fallback_seconds
        self.blocked_until = 0.0
        self.fallback_until = 0.0
        self.datacenter: object | None = None

    def _datacenter_session(self):
        if self.datacenter is None:
            proxy = settings.proxy.datacenter_proxy_url
            if not proxy:
                LOG.warning("No datacenter proxy configured; cannot fall back")
                return None
            self.datacenter = cffi.Session(
                impersonate="chrome", proxies={"http": proxy, "https": proxy}, verify=False
            )
        return self.datacenter

    def transport(self):
        """Own IP normally; the paid datacenter proxy for a window after a block."""
        if time.monotonic() < self.fallback_until:
            session = self._datacenter_session()
            if session is not None:
                return session, "datacenter"
        return self.session, "direct"

    def note_block(self) -> None:
        now = time.monotonic()
        self.blocked_until = now + self.block_pause_seconds
        self.fallback_until = now + self.fallback_seconds
        LOG.warning("Blocked; switching to datacenter proxy for %.0fs", self.fallback_seconds)

    def blocked_for(self) -> float:
        """Seconds the item-page worker should wait - zero while the proxy is covering us."""
        if time.monotonic() < self.fallback_until and self._datacenter_session() is not None:
            return 0.0
        return max(0.0, self.blocked_until - time.monotonic())

    def fetch_products(self, url: str, search_name: str) -> pd.DataFrame:
        session, label = self.transport()
        if label == "direct":
            self.pacer.wait()
        response = session.get(url, headers=HEADERS, timeout=40)
        if response.status_code != 200 and label == "direct":
            self.note_block()
            session, label = self.transport()
            if label == "datacenter":
                LOG.info("%s: retrying through the datacenter proxy", search_name)
                response = session.get(url, headers=HEADERS, timeout=40)
        if response.status_code != 200:
            LOG.warning("%s blocked/failed on %s: HTTP %s; no immediate retry",
                        search_name, label, response.status_code)
            return pd.DataFrame()
        products = requests_html.HTML(html=response.text).find(PRODUCT_SELECTOR)
        if not products:
            LOG.warning("%s returned no products; no immediate retry", search_name)
            return pd.DataFrame()
        rows = []
        for product in products:
            row = self.scraper.extract_catalog_item_meta(product, {}, 0, 0, get_images=True)
            if not row:
                continue
            row["SearchName"] = search_name
            row["Condition"] = parse_condition(product)
            row["ConditionStatusId"] = CONDITION_IDS.get(row["Condition"].casefold(), "")
            if str(row["Link"]).startswith("/"):
                row["Link"] = "https://www.vinted.it" + str(row["Link"])
            rows.append(row)
        return pd.DataFrame(rows).drop_duplicates("Dataid") if rows else pd.DataFrame()

    def search_page(self, name: str, config) -> pd.DataFrame:
        config.sort = "newest_first"
        return self.fetch_products(self.scraper.create_webpage(config) + "&page=1", name)

    def title_search(self, title: str, status_id: str, query: str | None = None) -> pd.DataFrame:
        query = (query or title).strip() or title
        rows = self.fetch_products(catalog_url(query, status_id, 1), query)
        if rows.empty:
            return rows
        expected = STATUS_NAMES[status_id]
        exact = rows["Condition"].fillna("").str.casefold().eq(expected.casefold())
        rejected = int((~exact).sum())
        if rejected:
            LOG.warning("Rejected %d title results with missing/wrong condition", rejected)
        rows = rows[exact].copy()
        rows["ConditionStatusId"] = status_id
        return rows


class JsonState:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"seen": [], "sent": [], "primed_searches": []}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            LOG.exception("Could not read state %s", self.path)
            return {"seen": [], "sent": [], "primed_searches": []}

    def values(self, key: str) -> set[str]:
        return {str(value) for value in self.data.get(key, [])}

    def add_many(self, key: str, values) -> None:
        self.data[key] = sorted(self.values(key) | {str(value) for value in values})
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def provisional_split(ranked: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    rows = ranked.copy()
    valid = (
        rows["title_similarity"].ge(config.title_min)
        & rows["image_similarity"].ge(config.image_min)
        & rows["combined_score"].ge(config.combined_min)
    )
    rows["decision"] = np.where(valid, "kept", "non_kept")
    rows["reason"] = np.where(
        valid,
        "passes provisional title + image similarity floors",
        "fails one or more provisional similarity floors",
    )
    return rows


def price_analysis(target_price: float, kept_prices: pd.Series, min_kept: int = 3) -> dict:
    prices = pd.to_numeric(kept_prices, errors="coerce")
    prices = prices[np.isfinite(prices) & prices.gt(0)]
    count = int(len(prices))
    if not count:
        return {"count": 0, "median": None, "discount_pct": None, "verdict": "No reliable comparable prices."}
    median = float(prices.median())
    discount = 100.0 * (median - target_price) / median
    if count < min_kept:
        verdict = f"Weak evidence: only {count} provisional exact comparable(s)."
    elif discount >= 25:
        verdict = "Possible deal: well below provisional comparable median."
    elif discount >= 10:
        verdict = "Below provisional comparable median."
    else:
        verdict = "Not clearly underpriced versus provisional comparables."
    return {"count": count, "median": median, "discount_pct": discount, "verdict": verdict}


def _score(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.3f}"


def write_mobile_report(
    path: Path,
    target: dict,
    rows: pd.DataFrame,
    candidates: pd.DataFrame,
    analysis: dict,
    musiq: float,
) -> None:
    merged = rows.merge(
        candidates[["Dataid", "Price", "Images", "Condition"]],
        left_on="candidate_item_id",
        right_on="Dataid",
        how="left",
    )

    def cards(decision: str) -> str:
        selected = merged[merged["decision"].eq(decision)]
        return "".join(
            f"""<article>
<img src="{html.escape(str(row['Images']))}" loading="lazy">
<h3>#{int(row['combined_rank'])} · {html.escape(str(row['candidate_title']))}</h3>
<p><b>€{float(row['Price']):.2f}</b> · {html.escape(str(row['Condition']))}</p>
<code>title {_score(row['title_similarity'])} · image {_score(row['image_similarity'])} · combined {_score(row['combined_score'])}</code>
<p>{html.escape(str(row['reason']))}</p>
<a href="{html.escape(str(row['listing_url']))}" target="_blank" rel="noopener">Open on Vinted</a>
</article>"""
            for _, row in selected.iterrows()
        )

    median = "—" if analysis["median"] is None else f"€{analysis['median']:.2f}"
    discount = "—" if analysis["discount_pct"] is None else f"{analysis['discount_pct']:+.1f}%"
    document = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(target['Title']))} comparison</title>
<style>
body{{font:15px system-ui;margin:14px;background:#f4f4f4;color:#222}}a{{color:#0645ad}}
.target,article{{background:white;border:1px solid #ddd;border-radius:10px;padding:12px}}
.target img,article img{{width:100%;height:240px;object-fit:contain;background:#eee;border-radius:7px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}
code{{display:block;font-size:12px;overflow-wrap:anywhere}}.warning{{background:#fff3cd;padding:12px;border-radius:8px}}
</style>
<h1>{html.escape(str(target['Title']))}</h1>
<section class="target"><img src="{html.escape(str(target['Images']))}">
<p><b>Target €{float(target['Price']):.2f}</b> · {html.escape(str(target['Condition']))}</p>
<p>MUSIQ {musiq:.1f} (0-100, lower is worse)</p>
<p><a href="{html.escape(str(target['Link']))}" target="_blank" rel="noopener">Open target on Vinted</a></p>
</section>
<p class="warning"><b>{html.escape(analysis['verdict'])}</b><br>
Provisional kept: {analysis['count']} · median {median} · target discount vs median {discount}.<br>
Matching uses only multilingual E5 title similarity and DINOv2 first-image similarity.
Thresholds are provisional, not probabilities. Asking prices, not sold prices.</p>
<h2>Provisionally kept ({int((merged['decision'] == 'kept').sum())})</h2><div class="grid">{cards('kept')}</div>
<h2>Non-kept ({int((merged['decision'] != 'kept').sum())})</h2><div class="grid">{cards('non_kept')}</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def report_host() -> str:
    """Prefer private Tailscale IP, then same-LAN IP."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        candidate = result.stdout.strip().splitlines()[0]
        if ipaddress.ip_address(candidate).is_private:
            return candidate
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return local_ip()


def start_report_server(root: Path, port: int) -> ThreadingHTTPServer:
    root.mkdir(parents=True, exist_ok=True)
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(root), **kwargs
    )
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _price_line(analysis: dict) -> str:
    median = "—" if analysis["median"] is None else f"€{analysis['median']:.2f}"
    discount = "—" if analysis["discount_pct"] is None else f"{analysis['discount_pct']:+.1f}%"
    return f"kept {analysis['count']} · median {median} · discount {discount}"


async def send_telegram(
    target: dict,
    analysis: dict,
    report_url: str,
    musiq: float,
    analysis_full: dict | None = None,
    breakdown: str = "",
) -> dict | None:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.constants import ParseMode

    token = settings.telegram.bot_token
    chat_id = settings.telegram.resolved_recommended_deals_chat_id
    if not token or chat_id is None:
        raise RuntimeError("Telegram bot token/chat is not configured")
    caption = (
        f"<b>Bad-photo candidate: {html.escape(str(target['Title']))}</b>\n"
        f"€{float(target['Price']):.2f} · {html.escape(str(target['Condition']))} · MUSIQ {musiq:.1f}\n"
        f"<b>Catalog data:</b> {html.escape(analysis['verdict'])}\n"
        f"{_price_line(analysis)}"
    )
    if analysis_full is not None:
        caption += (
            f"\n<b>Full data:</b> {html.escape(analysis_full['verdict'])}\n"
            f"{_price_line(analysis_full)}"
        )
    if breakdown:
        caption += f"\n<i>{html.escape(breakdown)}</i>"
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Vinted", url=str(target["Link"])),
        InlineKeyboardButton("Comparison", url=report_url),
    ]])
    bot = Bot(str(token))
    image = str(target.get("Images") or "")
    if image.startswith(("http://", "https://")):
        try:
            message = await bot.send_photo(chat_id=chat_id, photo=image, caption=caption, parse_mode=ParseMode.HTML, reply_markup=keyboard)
            return {"chat_id": chat_id, "message_id": message.message_id, "kind": "photo", "caption": caption,
                    "link": str(target["Link"]), "report_url": report_url}
        except Exception as exc:  # expired CDN signature, unreachable image, ...
            LOG.warning("Photo alert failed (%s); falling back to text", exc)
    message = await bot.send_message(chat_id=chat_id, text=caption, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    return {"chat_id": chat_id, "message_id": message.message_id, "kind": "text", "caption": caption,
            "link": str(target["Link"]), "report_url": report_url}


class Monitor:
    def __init__(self, args):
        self.args = args
        self.output = args.out_dir
        self.reports = self.output / "reports"
        self.state = JsonState(self.output / "state.json")
        self.client = CatalogClient(args.gap_seconds, args.catalog_block_pause, args.fallback_seconds)
        self.searches = load_searches(str(settings.paths.searches_yaml))
        missing = [name for name in SEARCHES if name not in self.searches]
        if missing:
            raise ValueError(f"missing search configs: {', '.join(missing)}")
        self.metric = None
        self.title_encoder = None
        self.image_encoder = None
        self.device = choose_device(args.device)
        self.queue_path = self.output / "full_compare_queue.json"
        self.queue_lock = threading.Lock()
        self.encoder_lock = threading.Lock()
        self.full_session = cffi.Session(impersonate="chrome")
        self.full_pacer = RequestPacer(args.full_gap_seconds, "item page")

    def score_musiq(self, image_path: Path) -> float:
        if self.metric is None:
            import pyiqa
            self.metric = pyiqa.create_metric("musiq", device=self.device)
        return float(self.metric(str(image_path)))

    def download_target(self, row: dict) -> Path | None:
        try:
            response = self.client.session.get(str(row["Images"]), headers=HEADERS, timeout=20)
            response.raise_for_status()
            suffix = Path(str(row["Images"]).split("?", 1)[0]).suffix or ".jpg"
            path = self.output / "target_images" / f"{row['Dataid']}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            with Image.open(path) as image:
                image.verify()
            return path
        except Exception:
            LOG.exception("Target image failed for item %s", row.get("Dataid"))
            return None

    def encoders(self):
        if self.title_encoder is None:
            self.title_encoder = E5TitleEncoder(self.args.title_model, self.device)
            self.image_encoder = DINOImageEncoder(self.args.image_model, self.device)
        return self.title_encoder, self.image_encoder

    def analyze(self, target: dict, target_image: Path, musiq: float) -> tuple[dict, str, pd.DataFrame, str]:
        status_id = str(target.get("ConditionStatusId") or "")
        if status_id not in STATUS_NAMES:
            candidates = pd.DataFrame(columns=["Dataid", "Price", "Images", "Condition"])
            analysis = price_analysis(float(target["Price"]), pd.Series(dtype=float))
            report = self.reports / f"{target['Dataid']}.html"
            write_mobile_report(report, target, pd.DataFrame(columns=[
                "candidate_item_id", "candidate_title", "listing_url", "title_similarity",
                "image_similarity", "combined_score", "combined_rank", "decision", "reason",
            ]), candidates, analysis, musiq)
            return analysis, report.name, candidates, "low"

        query = str(target["Title"])
        confidence = "unknown"
        if self.args.smart_query:
            rewrite = rewrite_title(query, self.output / "title_queries", brand=target.get("Brand"))
            if rewrite["query"] != query:
                LOG.info("Query rewritten: %r -> %r [%s]", query, rewrite["query"], rewrite["confidence"])
            query = rewrite["query"]
            confidence = rewrite["confidence"]
        candidates = self.client.title_search(str(target["Title"]), status_id, query=query)
        candidate_path = self.output / "matches" / str(target["Dataid"]) / "candidates.csv"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_path, index=False)
        if candidates.empty:
            ranked = pd.DataFrame(columns=[
                "candidate_item_id", "candidate_title", "listing_url", "title_similarity",
                "image_similarity", "combined_score", "combined_rank",
            ])
        else:
            with self.encoder_lock:  # the full-compare worker shares these models
                title_encoder, image_encoder = self.encoders()
                ranked = run_matching(
                    MatchConfig(
                        target_item_id=str(target["Dataid"]),
                        target_title=str(target["Title"]),
                        target_image=str(target_image),
                        candidates=candidate_path,
                        out_dir=candidate_path.parent / "matcher",
                        cache_dir=self.output / "embedding_cache",
                        top_k=self.args.top_k,
                        device=self.device,
                    ),
                    title_encoder,
                    image_encoder,
                )
        split = provisional_split(ranked, SplitConfig(
            self.args.title_min, self.args.image_min, self.args.combined_min,
            self.args.min_kept,
        ))
        kept_ids = set(split.loc[split["decision"].eq("kept"), "candidate_item_id"].astype(str))
        prices = candidates.loc[candidates["Dataid"].astype(str).isin(kept_ids), "Price"]
        analysis = price_analysis(float(target["Price"]), prices, self.args.min_kept)
        report = self.reports / f"{target['Dataid']}.html"
        write_mobile_report(report, target, split, candidates, analysis, musiq)
        split.to_csv(candidate_path.parent / "split.csv", index=False)
        return analysis, report.name, candidates, confidence

    def in_price_band(self, target_price: float, candidates: pd.DataFrame, target_id: str) -> list[str]:
        """Full-page fetches are expensive, so skip candidates priced absurdly far from the target.

        The title search returns the target itself, which run_matching also drops.
        """
        prices = pd.to_numeric(candidates["Price"], errors="coerce")
        band = self.args.price_band
        ids = candidates["Dataid"].astype(str)
        keep = prices.between(target_price / band, target_price * band) & ids.ne(str(target_id))
        return ids[keep].tolist()

    def _queue_read(self) -> list[dict]:
        if not self.queue_path.exists():
            return []
        try:
            jobs = json.loads(self.queue_path.read_text(encoding="utf-8"))
            return jobs if isinstance(jobs, list) else []
        except (OSError, json.JSONDecodeError):
            LOG.exception("Unreadable full-compare queue %s", self.queue_path)
            return []

    def _queue_write(self, jobs: list[dict]) -> None:
        """Worst photo first: the whole point of the queue is the badly shot listings."""
        jobs = sorted(jobs, key=lambda job: float(job.get("musiq", 0.0)))
        temporary = self.queue_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.queue_path)

    def enqueue_full_compare(self, job: dict) -> None:
        with self.queue_lock:
            jobs = self._queue_read()
            jobs.append(job)
            self._queue_write(jobs)
        LOG.info("Queued full comparison for %s at MUSIQ %.1f (%d waiting)",
                 job["item_id"], job["musiq"], len(jobs))

    def _queue_head(self) -> dict | None:
        with self.queue_lock:
            jobs = sorted(self._queue_read(), key=lambda job: float(job.get("musiq", 0.0)))
            return jobs[0] if jobs else None

    def _queue_finish(self, item_id: str, requeue: bool = False) -> None:
        with self.queue_lock:
            jobs = self._queue_read()
            remaining = [job for job in jobs if str(job.get("item_id")) != str(item_id)]
            done = [job for job in jobs if str(job.get("item_id")) == str(item_id)]
            if not done:
                return
            job = done[0]
            if requeue:
                job["attempts"] = int(job.get("attempts", 0)) + 1
                if job["attempts"] < MAX_FULL_COMPARE_ATTEMPTS:
                    remaining.append(job)
                else:
                    LOG.warning("Giving up on full comparison for %s after %d attempts",
                                item_id, job["attempts"])
            self._queue_write(remaining)

    def fetch_full_item(self, item_id: str) -> dict | None:
        """One item-page fetch: paced on our own IP, unpaced through the proxy."""
        while True:
            blocked = self.client.blocked_for()
            if blocked <= 0:
                break
            LOG.info("Catalog blocked; item-page worker waiting %.0fs", blocked)
            time.sleep(min(blocked, 60.0))

        session, label = self.client.transport()
        if label == "direct":
            session = self.full_session
            self.full_pacer.wait()

        def on_status(status: int) -> None:
            if status == 200:
                return
            LOG.warning("Item page %s on %s: HTTP %s", item_id, label, status)
            if label == "direct":
                self.client.note_block()

        return fetch_item(session, item_id, self.output / "full_items", HEADERS, on_status=on_status)

    def judge_specs(self, target_full: dict, fulls: list[dict], rows_b: pd.DataFrame) -> pd.DataFrame:
        """Let the model decide what is genuinely comparable; embeddings only shortlist."""
        verdicts = compare_specs(target_full, fulls, self.output / "spec_cache")
        if verdicts.empty:
            LOG.warning("No spec verdicts; keeping the embedding decision")
            return rows_b
        rows = rows_b.copy()
        rows["embedding_decision"] = rows["decision"]
        rows["candidate_item_id"] = rows["candidate_item_id"].astype(str)
        rows = rows.merge(verdicts, on="candidate_item_id", how="left")
        usable = set(usable_comparables(rows)["candidate_item_id"])
        rows["decision"] = np.where(rows["candidate_item_id"].isin(usable), "kept", "non_kept")
        rows["reason"] = np.where(
            rows["candidate_item_id"].isin(usable),
            "same product, full size, no disqualifier",
            rows["disqualifier"].fillna("not judged").replace("none", "different product")
            + rows["note"].fillna("").radd(": ").where(rows["note"].fillna("").ne(""), ""),
        )
        LOG.info("Spec pass: %d of %d candidates usable as comparables", len(usable), len(rows))
        return rows

    def run_full_compare(self, job: dict) -> bool:
        item_id = str(job["item_id"])
        matches = self.output / "matches" / item_id
        split_path = matches / "split.csv"
        if not split_path.exists():
            LOG.warning("No stored catalog split for %s; dropping job", item_id)
            return True

        target_full = self.fetch_full_item(item_id)
        if target_full is None:
            LOG.warning("Target page %s unavailable; will retry later", item_id)
            return False

        fulls = []
        for candidate_id in job["candidate_ids"]:
            data = self.fetch_full_item(str(candidate_id))
            if data is not None:
                fulls.append(data)
        LOG.info("%s: fetched %d/%d candidate pages", item_id, len(fulls), len(job["candidate_ids"]))
        if not fulls and job["candidate_ids"]:
            return False

        rows_b = pd.DataFrame(columns=["candidate_item_id", "decision", "price"])
        if fulls:
            with self.encoder_lock:
                title_encoder, image_encoder = self.encoders()
                rows_b = score_full(
                    target_full, fulls, title_encoder, image_encoder,
                    self.args.title_model, self.args.image_model,
                    cache_dir=self.output / "embedding_cache",
                    download_dir=matches / "full_photos",
                    config=FullSplitConfig(
                        photo_min=self.args.full_photo_min,
                        title_min=self.args.full_title_min,
                        combined_min=self.args.full_combined_min,
                        max_photos=self.args.max_photos,
                    ),
                )
        if self.args.spec_compare and not rows_b.empty:
            rows_b = self.judge_specs(target_full, fulls, rows_b)
        rows_b.to_csv(matches / "split_full.csv", index=False)

        target = job["target"]
        kept = rows_b.loc[rows_b["decision"].eq("kept"), "price"] if not rows_b.empty else pd.Series(dtype=float)
        analysis_b = price_analysis(float(target["Price"]), kept, self.args.min_kept)
        write_dual_report(
            self.reports / f"{item_id}.html", target, float(job["musiq"]),
            pd.read_csv(split_path), rows_b, job["analysis_a"], analysis_b,
        )
        breakdown = verdict_summary(rows_b) if self.args.spec_compare else ""
        LOG.info("%s: full comparison done (catalog kept %d, full kept %d) %s",
                 item_id, job["analysis_a"]["count"], analysis_b["count"], breakdown)

        # A title that names no product cannot produce comparables worth reading.
        if (job.get("title_confidence") == "low"
                and analysis_b["count"] < self.args.min_kept
                and self.args.skip_undecidable):
            LOG.info("%s: skipping alert - vague title %r and only %d comparable(s)",
                     item_id, job["target"]["Title"], analysis_b["count"])
            return True

        if not self.args.dry_run:
            asyncio.run(send_telegram(
                target, job["analysis_a"], job["report_url"], float(job["musiq"]), analysis_b,
                breakdown=breakdown,
            ))
            self.state.add_many("sent", [item_id])
        return True

    def full_compare_worker(self) -> None:
        while True:
            job = self._queue_head()
            if job is None:
                time.sleep(30)
                continue
            try:
                done = self.run_full_compare(job)
            except Exception:
                LOG.exception("Full comparison failed for %s", job.get("item_id"))
                done = False
            self._queue_finish(job["item_id"], requeue=not done)
            if not done:
                time.sleep(60)

    def process_search(self, name: str) -> None:
        page = self.client.search_page(name, self.searches[name])
        if page.empty:
            return
        ids = page["Dataid"].astype(str)
        if name not in self.state.values("primed_searches") and self.args.prime_first_cycle:
            self.state.add_many("seen", ids)
            self.state.add_many("primed_searches", [name])
            LOG.info("%s primed with %d current listings; alerts start next visit", name, len(page))
            return
        unseen = page[~ids.isin(self.state.values("seen"))].copy()
        LOG.info("%s: %d listings, %d unseen", name, len(page), len(unseen))
        for _, series in unseen.iterrows():
            target = series.to_dict()
            item_id = str(target["Dataid"])
            if item_id in self.state.values("sent"):
                self.state.add_many("seen", [item_id])
                continue
            image_path = self.download_target(target)
            if image_path is None:
                self.state.add_many("seen", [item_id])
                continue
            musiq = self.score_musiq(image_path)
            threshold = SEARCH_MUSIQ_THRESHOLDS.get(name, self.args.musiq_threshold)
            LOG.info("%s MUSIQ=%.1f (%s cut %.1f)", item_id, musiq, name, threshold)
            if musiq >= threshold:
                self.state.add_many("seen", [item_id])
                continue
            analysis, report_name, candidates, confidence = self.analyze(target, image_path, musiq)
            report_url = f"{self.args.public_base_url.rstrip('/')}/{quote(report_name)}"
            LOG.info("BAD %s | %s | %s", item_id, analysis["verdict"], report_url)
            self.state.add_many("seen", [item_id])
            if self.args.full_compare:
                # The alert waits for the full comparison; the worker sends it.
                self.enqueue_full_compare({
                    "item_id": item_id,
                    "musiq": musiq,
                    "report_url": report_url,
                    "analysis_a": analysis,
                    "candidate_ids": self.in_price_band(float(target["Price"]), candidates, item_id)
                    if not candidates.empty else [],
                    "target": {key: target[key] for key in ("Title", "Price", "Condition", "Link", "Images")},
                    "title_confidence": confidence,
                    "attempts": 0,
                })
            elif not self.args.dry_run:
                asyncio.run(send_telegram(target, analysis, report_url, musiq))
                self.state.add_many("sent", [item_id])

    def run(self) -> None:
        if self.args.full_compare:
            threading.Thread(target=self.full_compare_worker, daemon=True).start()
        while True:
            for name in SEARCHES:
                self.process_search(name)
                if self.args.once:
                    return


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE / "data" / "deal_monitor")
    parser.add_argument("--gap-seconds", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--public-base-url", default="")
    parser.add_argument("--musiq-threshold", type=float, default=MUSIQ_BAD_THRESHOLD)
    parser.add_argument("--title-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--image-model", default="facebook/dinov2-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--title-min", type=float, default=0.82)
    parser.add_argument("--image-min", type=float, default=0.55)
    parser.add_argument("--combined-min", type=float, default=0.70)
    parser.add_argument("--min-kept", type=int, default=3)
    parser.add_argument("--full-gap-seconds", type=float, default=90.0)
    parser.add_argument("--catalog-block-pause", type=float, default=600.0)
    parser.add_argument("--price-band", type=float, default=10.0,
                        help="skip candidates priced beyond this multiple of the target price")
    parser.add_argument("--max-photos", type=int, default=8)
    parser.add_argument("--full-photo-min", type=float, default=0.60)
    parser.add_argument("--full-title-min", type=float, default=0.75)
    parser.add_argument("--full-combined-min", type=float, default=0.68)
    parser.add_argument("--fallback-seconds", type=float, default=600.0,
                        help="how long to route through the datacenter proxy after a block")
    parser.add_argument("--no-full-compare", dest="full_compare", action="store_false")
    parser.add_argument("--no-smart-query", dest="smart_query", action="store_false",
                        help="search with the raw listing title instead of the rewritten query")
    parser.add_argument("--no-spec-compare", dest="spec_compare", action="store_false",
                        help="decide comparables from embeddings alone, without the model")
    parser.add_argument("--no-skip-undecidable", dest="skip_undecidable", action="store_false",
                        help="alert even when the title names no product and comparables are too few")
    parser.set_defaults(full_compare=True, smart_query=True, spec_compare=True, skip_undecidable=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-prime-first-cycle", dest="prime_first_cycle", action="store_false")
    parser.set_defaults(prime_first_cycle=True)
    args = parser.parse_args()
    if args.gap_seconds < 0 or args.top_k < 1 or args.min_kept < 1:
        parser.error("gap must be >= 0; top-k and min-kept must be >= 1")
    if args.full_gap_seconds < 0 or args.max_photos < 1 or args.price_band <= 1:
        parser.error("full-gap must be >= 0; max-photos >= 1; price-band > 1")
    for value in (args.title_min, args.image_min, args.combined_min):
        if not -1 <= value <= 1:
            parser.error("similarity floors must be between -1 and 1")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_report_server(args.out_dir / "reports", args.port)
    if not args.public_base_url:
        args.public_base_url = f"http://{report_host()}:{args.port}"
    LOG.info("Reports: %s", args.public_base_url)
    Monitor(args).run()


if __name__ == "__main__":
    main()
