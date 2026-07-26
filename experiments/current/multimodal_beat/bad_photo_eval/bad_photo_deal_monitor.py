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
MUSIQ_BAD_THRESHOLD = 67.7


@dataclass(frozen=True)
class SplitConfig:
    title_min: float = 0.82
    image_min: float = 0.55
    combined_min: float = 0.70
    min_kept_for_verdict: int = 3


class RequestPacer:
    def __init__(self, gap_seconds: float):
        self.gap_seconds = gap_seconds
        self.last_request_at: float | None = None

    def wait(self) -> None:
        if self.last_request_at is not None:
            remaining = self.gap_seconds - (time.monotonic() - self.last_request_at)
            if remaining > 0:
                LOG.info("Chill pace: waiting %.1fs before next catalog request", remaining)
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


class CatalogClient:
    def __init__(self, gap_seconds: float):
        self.session = cffi.Session(impersonate="chrome")
        self.pacer = RequestPacer(gap_seconds)
        self.scraper = Simple_scraper()

    def fetch_products(self, url: str, search_name: str) -> pd.DataFrame:
        self.pacer.wait()
        response = self.session.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            LOG.warning("%s blocked/failed: HTTP %s; no immediate retry", search_name, response.status_code)
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

    def title_search(self, title: str, status_id: str) -> pd.DataFrame:
        rows = self.fetch_products(catalog_url(title, status_id, 1), title)
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
<p>MUSIQ {musiq:.1f} (bad when &lt; {MUSIQ_BAD_THRESHOLD:.1f})</p>
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


async def send_telegram(target: dict, analysis: dict, report_url: str, musiq: float) -> None:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.constants import ParseMode

    token = settings.telegram.bot_token
    chat_id = settings.telegram.resolved_recommended_deals_chat_id
    if not token or chat_id is None:
        raise RuntimeError("Telegram bot token/chat is not configured")
    median = "—" if analysis["median"] is None else f"€{analysis['median']:.2f}"
    discount = "—" if analysis["discount_pct"] is None else f"{analysis['discount_pct']:+.1f}%"
    caption = (
        f"<b>Bad-photo candidate: {html.escape(str(target['Title']))}</b>\n"
        f"€{float(target['Price']):.2f} · {html.escape(str(target['Condition']))} · MUSIQ {musiq:.1f}\n"
        f"{html.escape(analysis['verdict'])}\n"
        f"Kept {analysis['count']} · median {median} · discount {discount}"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Vinted", url=str(target["Link"])),
        InlineKeyboardButton("Comparison", url=report_url),
    ]])
    bot = Bot(str(token))
    image = str(target.get("Images") or "")
    if image.startswith(("http://", "https://")):
        await bot.send_photo(chat_id=chat_id, photo=image, caption=caption, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    else:
        await bot.send_message(chat_id=chat_id, text=caption, parse_mode=ParseMode.HTML, reply_markup=keyboard)


class Monitor:
    def __init__(self, args):
        self.args = args
        self.output = args.out_dir
        self.reports = self.output / "reports"
        self.state = JsonState(self.output / "state.json")
        self.client = CatalogClient(args.gap_seconds)
        self.searches = load_searches(str(settings.paths.searches_yaml))
        missing = [name for name in SEARCHES if name not in self.searches]
        if missing:
            raise ValueError(f"missing search configs: {', '.join(missing)}")
        self.metric = None
        self.title_encoder = None
        self.image_encoder = None
        self.device = choose_device(args.device)

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

    def analyze(self, target: dict, target_image: Path, musiq: float) -> tuple[dict, str]:
        status_id = str(target.get("ConditionStatusId") or "")
        if status_id not in STATUS_NAMES:
            candidates = pd.DataFrame(columns=["Dataid", "Price", "Images", "Condition"])
            analysis = price_analysis(float(target["Price"]), pd.Series(dtype=float))
            report = self.reports / f"{target['Dataid']}.html"
            write_mobile_report(report, target, pd.DataFrame(columns=[
                "candidate_item_id", "candidate_title", "listing_url", "title_similarity",
                "image_similarity", "combined_score", "combined_rank", "decision", "reason",
            ]), candidates, analysis, musiq)
            return analysis, report.name

        candidates = self.client.title_search(str(target["Title"]), status_id)
        candidate_path = self.output / "matches" / str(target["Dataid"]) / "candidates.csv"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_path, index=False)
        if candidates.empty:
            ranked = pd.DataFrame(columns=[
                "candidate_item_id", "candidate_title", "listing_url", "title_similarity",
                "image_similarity", "combined_score", "combined_rank",
            ])
        else:
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
        return analysis, report.name

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
            LOG.info("%s MUSIQ=%.1f", item_id, musiq)
            if musiq >= self.args.musiq_threshold:
                self.state.add_many("seen", [item_id])
                continue
            analysis, report_name = self.analyze(target, image_path, musiq)
            report_url = f"{self.args.public_base_url.rstrip('/')}/{quote(report_name)}"
            LOG.info("BAD %s | %s | %s", item_id, analysis["verdict"], report_url)
            if not self.args.dry_run:
                asyncio.run(send_telegram(target, analysis, report_url, musiq))
                self.state.add_many("sent", [item_id])
            self.state.add_many("seen", [item_id])

    def run(self) -> None:
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
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-prime-first-cycle", dest="prime_first_cycle", action="store_false")
    parser.set_defaults(prime_first_cycle=True)
    args = parser.parse_args()
    if args.gap_seconds < 0 or args.top_k < 1 or args.min_kept < 1:
        parser.error("gap must be >= 0; top-k and min-kept must be >= 1")
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
