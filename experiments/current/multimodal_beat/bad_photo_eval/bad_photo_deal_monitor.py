"""Slow local bad-photo deal monitor for Vinted.

One Vinted catalog request is made at most once per minute. New listings with
MUSIQ below the configured threshold get a condition-locked search built from
the item's own page, a comparison over every photo and the description of both
sides, a spec verdict from gpt-5-nano, a mobile HTML report and a Telegram alert.
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
    choose_device,
)
from full_compare import FullSplitConfig, candidate_frame, score_images, write_full_report  # noqa: E402
from full_item_fetch import cached_item, fetch_item  # noqa: E402
from image_compare import compare_photos  # noqa: E402
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
        if not (response.text or "").strip():
            LOG.warning("%s returned an empty body on %s", search_name, label)
            return pd.DataFrame()
        try:
            products = requests_html.HTML(html=response.text).find(PRODUCT_SELECTOR)
        except Exception:
            LOG.exception("%s returned unparseable HTML", search_name)
            return pd.DataFrame()
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


def _deal_mark(analysis: dict) -> str:
    """Green only when the discount is large enough to be worth opening."""
    discount = analysis.get("discount_pct")
    if discount is None or not analysis.get("count"):
        return "⚫"
    if discount >= 25:
        return "🟢"
    if discount >= 10:
        return "🟡"
    return "⚪"


def _verdict_block(label: str, analysis: dict) -> str:
    median = "—" if analysis["median"] is None else f"<b>€{analysis['median']:.2f}</b>"
    discount = "—" if analysis["discount_pct"] is None else f"<b>{analysis['discount_pct']:+.1f}%</b>"
    return (f"{_deal_mark(analysis)} <b>{label}</b> · {html.escape(analysis['verdict'])}\n"
            f"      {analysis['count']} comparabili · mediana {median} · sconto {discount}")


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
    headline = analysis_full if analysis_full is not None else analysis
    caption = (
        f"{_deal_mark(headline)} <b>{html.escape(str(target['Title']))}</b>\n"
        f"💶 <b>€{float(target['Price']):.2f}</b> · {html.escape(str(target['Condition']))}"
        f" · 📷 MUSIQ <b>{musiq:.1f}</b>\n\n"
        f"{_verdict_block('Solo catalogo', analysis)}"
    )
    if analysis_full is not None:
        caption += f"\n{_verdict_block('Dati completi', analysis_full)}"
    if breakdown:
        caption += f"\n\n🔎 <i>{html.escape(breakdown)}</i>"
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("🛒 Vinted", url=str(target["Link"])),
        InlineKeyboardButton("📊 Confronto", url=report_url),
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

    def find_candidates(self, target: dict) -> tuple[pd.DataFrame, str, str]:
        """Search Vinted for listings of the same product. No comparison happens here."""
        status_id = str(target.get("ConditionStatusId") or "")
        if status_id not in STATUS_NAMES:
            return pd.DataFrame(), "low", ""

        query = str(target["Title"])
        confidence = "unknown"
        rewrite: dict = {}
        if self.args.smart_query:
            # Lazy titles often name the product only in the description, so read the item
            # page before searching. It is cached, so the full-compare worker reuses it.
            page = self.fetch_full_item(str(target["Dataid"])) or {}
            rewrite = rewrite_title(query, self.output / "title_queries",
                                    brand=page.get("brand") or target.get("Brand"),
                                    category=page.get("category"),
                                    description=page.get("description"))
            if rewrite["query"] != query:
                LOG.info("Query rewritten: %r -> %r [%s]", query, rewrite["query"], rewrite["confidence"])
            query = rewrite["query"]
            confidence = rewrite["confidence"]

        # Sellers here write in both languages, so search both and merge on item id.
        queries = [query]
        if self.args.dual_search and rewrite.get("query_local"):
            queries.append(rewrite["query_local"].strip())
        # The material is not in the query because it over-narrows some categories, but for
        # clothing it is the only word that matters: "Polo Ralph Lauren shirt" returned 18
        # cotton shirts and 0 comparables, "...linen shirt" returned 18 linen ones and 7.
        # Searching both ways and merging gets the breadth and the precision.
        material = (rewrite.get("material") or "").strip() if self.args.smart_query else ""
        if material and material.casefold() not in query.casefold():
            product_type = (rewrite.get("product_type") or "").strip()
            if product_type and query.casefold().endswith(product_type.casefold()):
                head = query[: -len(product_type)].rstrip()
                queries.append(f"{head} {material} {product_type}".strip())
            else:
                queries.append(f"{query} {material}".strip())
        frames = []
        for text in dict.fromkeys(q for q in queries if q):
            found = self.client.title_search(str(target["Title"]), status_id, query=text)
            LOG.info("%s: %d rows for %r", target["Dataid"], len(found), text)
            if not found.empty:
                found = found.copy()
                found["QueryRank"] = range(len(found))
                frames.append(found)

        # Interleave the queries rather than concatenating them: whatever cap applies later
        # would otherwise be filled entirely by the first, broadest query.
        candidates = pd.DataFrame()
        if frames:
            candidates = (pd.concat(frames, ignore_index=True)
                          .sort_values("QueryRank", kind="stable")
                          .drop_duplicates("Dataid")
                          .reset_index(drop=True))

        candidate_path = self.output / "matches" / str(target["Dataid"]) / "candidates.csv"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_path, index=False)
        LOG.info("%s: %d distinct candidates", target["Dataid"], len(candidates))
        return candidates, confidence, (rewrite.get("material") or "")

    def in_price_band(self, target_price: float, candidates: pd.DataFrame, target_id: str) -> list[str]:
        """Full-page fetches are expensive, so skip candidates priced absurdly far from the target.

        The title search returns the target itself, which run_matching also drops.
        """
        prices = pd.to_numeric(candidates["Price"], errors="coerce")
        band = self.args.price_band
        ids = candidates["Dataid"].astype(str)
        keep = prices.between(target_price / band, target_price * band) & ids.ne(str(target_id))
        chosen = ids[keep].tolist()
        if len(chosen) > self.args.max_candidates:
            LOG.info("Capping %d in-band candidates to %d", len(chosen), self.args.max_candidates)
            chosen = chosen[: self.args.max_candidates]
        return chosen

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
        cached = cached_item(item_id, self.output / "full_items")
        if cached is not None:  # nothing to pace, no request happens
            return cached
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

    def judge_specs(self, target_full: dict, fulls: list[dict], rows: pd.DataFrame) -> pd.DataFrame:
        """The model reads titles and descriptions and decides what is comparable."""
        verdicts = compare_specs(target_full, fulls, self.output / "spec_cache")
        rows = rows.copy()
        if verdicts.empty:
            LOG.warning("No spec verdicts; treating every candidate as unjudged")
            rows["decision"] = "non_kept"
            rows["reason"] = "not judged"
            return rows
        rows["candidate_item_id"] = rows["candidate_item_id"].astype(str)
        rows = rows.merge(verdicts, on="candidate_item_id", how="left")
        usable = set(usable_comparables(rows)["candidate_item_id"])
        rows["decision"] = np.where(rows["candidate_item_id"].isin(usable), "kept", "non_kept")
        rows["reason"] = np.where(
            rows["candidate_item_id"].isin(usable),
            "same product, comparable spec",
            rows["disqualifier"].fillna("not judged").replace("none", "different product")
            + rows["note"].fillna("").radd(": ").where(rows["note"].fillna("").ne(""), ""),
        )
        LOG.info("Text verdict: %d of %d candidates comparable", len(usable), len(rows))
        return rows

    def judge_photos(self, target_full: dict, survivors: list[dict], rows: pd.DataFrame) -> pd.DataFrame:
        """Second opinion on the photos, kept alongside the DINO cosine rather than replacing it."""
        verdicts = []
        for candidate in survivors:
            answer = compare_photos(target_full, candidate, self.output / "image_cache")
            if answer:
                verdicts.append({"candidate_item_id": str(candidate["item_id"]),
                                 "photo_same_product": answer.get("same_product"),
                                 "photo_same_colour": answer.get("same_colour"),
                                 "photo_confidence": answer.get("confidence"),
                                 "photo_note": answer.get("note")})
        if not verdicts:
            return rows
        rows = rows.copy()
        rows["candidate_item_id"] = rows["candidate_item_id"].astype(str)
        rows = rows.merge(pd.DataFrame(verdicts), on="candidate_item_id", how="left")
        judged = rows.dropna(subset=["photo_same_product"])
        if not judged.empty:
            agree = judged["photo_same_product"].astype(bool)
            LOG.info("Photo model says same product for %d of %d judged (DINO mean %.3f for yes, %.3f for no)",
                     int(agree.sum()), len(judged),
                     judged.loc[agree, "photo_similarity"].mean() if agree.any() else float("nan"),
                     judged.loc[~agree, "photo_similarity"].mean() if (~agree).any() else float("nan"))
        return rows

    def run_full_compare(self, job: dict) -> bool:
        item_id = str(job["item_id"])
        matches = self.output / "matches" / item_id
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

        rows = candidate_frame(target_full, fulls)
        if job.get("material"):
            target_full = {**target_full, "material": job["material"]}
        if rows.empty:
            rows = pd.DataFrame(columns=["candidate_item_id", "decision", "price", "reason"])
        elif self.args.spec_compare:
            rows = self.judge_specs(target_full, fulls, rows)
        else:
            rows["decision"] = "kept"
            rows["reason"] = "text verdict disabled"

        # Photos are only worth downloading for candidates the text already accepted.
        survivors = [item for item in fulls
                     if str(item["item_id"]) in set(rows.loc[rows["decision"].eq("kept"), "candidate_item_id"])]
        if survivors:
            with self.encoder_lock:
                _, image_encoder = self.encoders()
                photos = score_images(
                    target_full, survivors, image_encoder, self.args.image_model,
                    cache_dir=self.output / "embedding_cache",
                    download_dir=matches / "full_photos",
                    config=FullSplitConfig(max_photos=self.args.max_photos),
                )
            rows = rows.merge(photos, on="candidate_item_id", how="left")
            LOG.info("Compared photos for %d comparable candidates", len(survivors))
            if self.args.image_llm:
                rows = self.judge_photos(target_full, survivors, rows)
        rows_b = rows
        rows_b.to_csv(matches / "split_full.csv", index=False)

        target = job["target"]
        kept = rows_b.loc[rows_b["decision"].eq("kept"), "price"] if not rows_b.empty else pd.Series(dtype=float)
        analysis = price_analysis(float(target["Price"]), kept, self.args.min_kept)
        write_full_report(self.reports / f"{item_id}.html", target, float(job["musiq"]), rows_b, analysis)
        breakdown = verdict_summary(rows_b) if self.args.spec_compare else ""
        LOG.info("%s: comparison done (%d comparable of %d) %s",
                 item_id, analysis["count"], len(rows_b), breakdown)

        # A title that names no product cannot produce comparables worth reading.
        if (job.get("title_confidence") == "low"
                and analysis["count"] < self.args.min_kept
                and self.args.skip_undecidable):
            LOG.info("%s: skipping alert - vague title %r and only %d comparable(s)",
                     item_id, job["target"]["Title"], analysis["count"])
            return True

        if not self.args.dry_run:
            asyncio.run(send_telegram(
                target, analysis, job["report_url"], float(job["musiq"]), breakdown=breakdown,
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
            candidates, confidence, material = self.find_candidates(target)
            report_url = f"{self.args.public_base_url.rstrip('/')}/{quote(item_id)}.html"
            LOG.info("BAD %s | MUSIQ %.1f | %s", item_id, musiq, report_url)
            self.state.add_many("seen", [item_id])
            self.enqueue_full_compare({
                "item_id": item_id,
                "musiq": musiq,
                "report_url": report_url,
                "candidate_ids": self.in_price_band(float(target["Price"]), candidates, item_id)
                if not candidates.empty else [],
                "target": {key: target[key] for key in ("Title", "Price", "Condition", "Link", "Images")},
                "title_confidence": confidence,
                "material": material,
                "attempts": 0,
            })

    def run(self) -> None:
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
    parser.add_argument("--min-kept", type=int, default=3)
    parser.add_argument("--full-gap-seconds", type=float, default=90.0)
    parser.add_argument("--catalog-block-pause", type=float, default=600.0)
    parser.add_argument("--max-candidates", type=int, default=40,
                        help="most candidates to fetch full pages for, per alert")
    parser.add_argument("--price-band", type=float, default=10.0,
                        help="skip candidates priced beyond this multiple of the target price")
    parser.add_argument("--max-photos", type=int, default=8)
    parser.add_argument("--full-photo-min", type=float, default=0.60)
    parser.add_argument("--full-title-min", type=float, default=0.75)
    parser.add_argument("--full-combined-min", type=float, default=0.68)
    parser.add_argument("--fallback-seconds", type=float, default=600.0,
                        help="how long to route through the datacenter proxy after a block")
    parser.add_argument("--no-smart-query", dest="smart_query", action="store_false",
                        help="search with the raw listing title instead of the rewritten query")
    parser.add_argument("--no-spec-compare", dest="spec_compare", action="store_false",
                        help="decide comparables from embeddings alone, without the model")
    parser.add_argument("--no-dual-search", dest="dual_search", action="store_false",
                        help="search only the English query instead of English and Italian")
    parser.add_argument("--no-image-llm", dest="image_llm", action="store_false",
                        help="rank photos with DINO only, without asking the model")
    parser.add_argument("--no-skip-undecidable", dest="skip_undecidable", action="store_false",
                        help="alert even when the title names no product and comparables are too few")
    parser.set_defaults(smart_query=True, spec_compare=True, skip_undecidable=True,
                        dual_search=True, image_llm=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-prime-first-cycle", dest="prime_first_cycle", action="store_false")
    parser.set_defaults(prime_first_cycle=True)
    args = parser.parse_args()
    if args.gap_seconds < 0 or args.top_k < 1 or args.min_kept < 1:
        parser.error("gap must be >= 0; top-k and min-kept must be >= 1")
    if args.full_gap_seconds < 0 or args.max_photos < 1 or args.price_band <= 1:
        parser.error("full-gap must be >= 0; max-photos >= 1; price-band > 1")
    for value in (args.full_photo_min, args.full_title_min, args.full_combined_min):
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
