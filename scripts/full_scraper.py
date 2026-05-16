from Scraper import Scraper
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import pandas as pd
import utils_lib.utils as utils
from datetime import datetime
import multiprocessing
import tracemalloc
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_SWEEP_RUN = Path("data/experiments/deal_finder/offline_runs/sweep_20260510_222252")
DEFAULT_LOW_SCORE_THRESHOLD = 0.05
DEFAULT_HIGH_SCORE_THRESHOLD = 0.95
FULL_SCRAPE_DIR_NAME = "full_scrape"
FULL_SCRAPE_ITEMS_FILENAME = "items_enriched.csv"
FULL_SCRAPE_FAILURES_FILENAME = "full_scrape_failures.csv"
FULL_SCRAPE_EVENTS_FILENAME = "full_scrape_events.csv"
SCORE_EXTREME_REASONS = {"score_low", "score_high"}


def scrape_worker(args):
                time.sleep(15)
                """Top‐level for pickling: calls scrape_single_product and tacks its data onto the meta."""
                scraper, meta = args
                new_row, new_seller = scraper.scrape_single_product(
                    url=meta["Link"],
                    data_id=meta["Dataid"],
                    get_images=meta["get_images"]
                )

                # merge into full row
                meta.update({
                    "Images":           new_row.get("Images", []),
                    "Interested_count": new_row.get("Interested_count", 0),
                    "View_count":        new_row.get("View_count", 0),
                    "Description":  new_row.get("Description", ""),
                    "Condition":         new_row.get("Condition", ""),
                    # "Dataid":          new_row.get("Dataid", ""),   
                    "SearchDate":        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "MarketStatus":      "On Sale",
                    "Upload_date":       new_row.get("Upload_date", ""),
                    "SellerName":        new_seller.get("SellerName", ""),
                    "SellerId":          new_seller.get("SellerId", ""),
                    "Location":         new_seller.get("Location", ""),
                    "ReviewsCount":    new_seller.get("ReviewsCount", 0),
                    "Stars":           new_seller.get("Stars", -1),
                    "PrimaryImageUrl":  new_row.get("PrimaryImageUrl", ""),
                    "FullImageUrls":    new_row.get("FullImageUrls", []),
                    "VisiblePictureCount": new_row.get("VisiblePictureCount", 0),
                    "HiddenPictureCount": new_row.get("HiddenPictureCount", 0),
                    "PictureCount":     new_row.get("PictureCount", 0),
                    # "Page":              page+1,
                    # "SearchCount":       search_count,
                })

                seller_data = {
                    "SellerId": meta["SellerId"],
                    "SellerName": meta["SellerName"],
                    "Location": meta["Location"],
                    "ReviewsCount": meta["ReviewsCount"],
                    "Stars": meta["Stars"]
                }
                return meta, seller_data

class Full_Scraper(Scraper):
    def __init__(self):
        super().__init__()

    def _simple_scrape_root(self) -> Path:
        try:
            from config.project_config import settings

            return Path(str(settings.paths.simple_scrape_dir))
        except Exception:
            return Path("data/simple_scrape")

    def _full_scrape_dir(self, search_name: str, output_subdir: str | None = None) -> Path:
        subdir = str(output_subdir or FULL_SCRAPE_DIR_NAME).strip()
        if not subdir:
            subdir = FULL_SCRAPE_DIR_NAME
        return self._simple_scrape_root() / str(search_name) / subdir

    def _full_scrape_paths(self, search_name: str, output_subdir: str | None = None) -> dict[str, Path]:
        out_dir = self._full_scrape_dir(search_name, output_subdir=output_subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "items": out_dir / FULL_SCRAPE_ITEMS_FILENAME,
            "failures": out_dir / FULL_SCRAPE_FAILURES_FILENAME,
            "events": out_dir / FULL_SCRAPE_EVENTS_FILENAME,
        }

    def _read_csv_or_empty(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            self.logger.warning("Failed reading %s: %s: %s", path, type(exc).__name__, exc)
            return pd.DataFrame()

    def _write_csv_atomic(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)

    def _identity_key(self, row: pd.Series | dict) -> str:
        data_id = row.get("Dataid") if isinstance(row, dict) else row.get("Dataid")
        if data_id is not None and not (isinstance(data_id, float) and np.isnan(data_id)):
            text = str(data_id).strip()
            if text.endswith(".0"):
                text = text[:-2]
            if text:
                return f"Dataid:{text}"
        link = row.get("Link") if isinstance(row, dict) else row.get("Link")
        text = str(link or "").strip()
        return f"Link:{text}" if text else ""

    def _identity_keys_for_frame(self, df: pd.DataFrame) -> set[str]:
        if df.empty:
            return set()
        return {key for key in (self._identity_key(row) for _, row in df.iterrows()) if key}

    def _dedupe_by_identity(self, df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        out["_FullScrapeIdentity"] = [self._identity_key(row) for _, row in out.iterrows()]
        with_key = out[out["_FullScrapeIdentity"].astype(str).str.len() > 0].copy()
        without_key = out[out["_FullScrapeIdentity"].astype(str).str.len() == 0].copy()
        if not with_key.empty:
            with_key = with_key.drop_duplicates(subset=["_FullScrapeIdentity"], keep=keep)
        return pd.concat([with_key, without_key], ignore_index=True).drop(columns=["_FullScrapeIdentity"], errors="ignore")

    def _append_csv(self, path: Path, rows: list[dict], *, dedupe: bool = False) -> None:
        if not rows:
            return
        incoming = pd.DataFrame(rows)
        existing = self._read_csv_or_empty(path)
        combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
        if dedupe:
            combined = self._dedupe_by_identity(combined, keep="last")
        self._write_csv_atomic(combined, path)

    def _clean_image_url(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = text.replace("\\/", "/").replace("&amp;", "&")
        if text.startswith("//"):
            text = "https:" + text
        return text

    def _append_unique_image_url(self, urls: list[str], value: Any) -> None:
        text = self._clean_image_url(value)
        if not text or not text.startswith(("http://", "https://")):
            return
        if text not in urls:
            urls.append(text)

    def _image_urls_from_value(self, value: Any) -> list[str]:
        out: list[str] = []
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return out
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                    return self._image_urls_from_value(parsed)
                except Exception:
                    pass
            self._append_unique_image_url(out, text)
            return out
        if isinstance(value, dict):
            for key in ("url", "src", "full_size_url", "image_url", "thumbnail_url"):
                if key in value:
                    self._append_unique_image_url(out, value.get(key))
            for child in value.values():
                for url in self._image_urls_from_value(child):
                    self._append_unique_image_url(out, url)
            return out
        if isinstance(value, (list, tuple, set)):
            for item in value:
                for url in self._image_urls_from_value(item):
                    self._append_unique_image_url(out, url)
        return out

    def _hidden_picture_count_from_text(self, value: Any) -> int:
        text = str(value or "").strip()
        if not text:
            return 0
        text = (
            text.replace("\xa0", " ")
            .replace("&nbsp;", " ")
            .replace("＋", "+")
            .strip()
        )
        match = re.search(
            r"(?:^|[\s>])(?:\+|&plus;)\s*(\d{1,3})(?:\s*(?:foto|photo|photos|immagini|images))?(?:$|[\s<])",
            text,
            flags=re.IGNORECASE,
        )
        return int(match.group(1)) if match else 0

    def _hidden_picture_count_from_elements(
        self,
        elements: Any,
        *,
        _seen: set[int] | None = None,
        _depth: int = 0,
        _max_depth: int = 4,
    ) -> int:
        if elements is None:
            return 0
        if not isinstance(elements, (list, tuple, set)):
            elements = [elements]
        if _seen is None:
            _seen = set()
        hidden_count = 0
        for element in elements:
            element_id = id(element)
            if element_id in _seen:
                continue
            _seen.add(element_id)
            text = getattr(element, "text", "") or ""
            hidden_count = max(hidden_count, self._hidden_picture_count_from_text(text))
            attrs = getattr(element, "attrs", {}) or {}
            for attr in ("aria-label", "title", "data-testid", "data-test-id"):
                hidden_count = max(hidden_count, self._hidden_picture_count_from_text(attrs.get(attr)))
            if _depth >= _max_depth:
                continue
            try:
                children = element.find("*")
            except Exception:
                children = []
            if children:
                hidden_count = max(
                    hidden_count,
                    self._hidden_picture_count_from_elements(
                        children,
                        _seen=_seen,
                        _depth=_depth + 1,
                        _max_depth=_max_depth,
                    ),
                )
        return hidden_count

    def extract_item_page_image_metadata(self, html_content, fallback_primary_url: Any = None) -> dict[str, Any]:
        urls: list[str] = []
        visible_urls: list[str] = []

        selector_groups = [
            'div[class*="item-photos"] img',
            'button[class*="item-thumbnail"] img',
            '[data-testid*="item-photo"] img',
            '[data-testid*="item-photo"]',
        ]
        for selector in selector_groups:
            try:
                elements = html_content.find(selector)
            except Exception:
                elements = []
            for element in elements:
                attrs = getattr(element, "attrs", {}) or {}
                for attr in ("src", "data-src", "content"):
                    if attr in attrs:
                        self._append_unique_image_url(urls, attrs.get(attr))
                        self._append_unique_image_url(visible_urls, attrs.get(attr))

        for selector, attr in (
            ('meta[property="og:image"]', "content"),
            ('meta[name="twitter:image"]', "content"),
            ('meta[itemprop="image"]', "content"),
        ):
            try:
                element = html_content.find(selector, first=True)
            except Exception:
                element = None
            if element:
                self._append_unique_image_url(urls, getattr(element, "attrs", {}).get(attr))

        try:
            scripts = html_content.find('script[type="application/ld+json"]')
        except Exception:
            scripts = []
        for script in scripts:
            raw_json = getattr(script, "text", None) or ""
            if not raw_json.strip():
                continue
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            for obj in self._iter_json_dicts(payload):
                if isinstance(obj, dict) and "image" in obj:
                    for url in self._image_urls_from_value(obj.get("image")):
                        self._append_unique_image_url(urls, url)

        if not urls:
            html_text = getattr(html_content, "html", "") or ""
            for match in re.findall(r"https?:\\?/\\?/[^\"'<>\\\s]+?\.(?:webp|jpg|jpeg|png)(?:\?[^\"'<>\\\s]*)?", html_text, flags=re.IGNORECASE):
                self._append_unique_image_url(urls, match)

        for url in self._image_urls_from_value(fallback_primary_url):
            self._append_unique_image_url(urls, url)

        hidden_count = 0
        hidden_source_selectors = [
            'div[class*="item-photos"]',
            'div[class*="item-photos"] button',
            'div[class*="item-photos"] button div',
            'section figure button',
            'section figure button div',
            'main figure button',
            'main figure button div',
            'figure button',
            'figure button div',
            'button[aria-label*="photo"]',
            'button[aria-label*="foto"]',
            'button[aria-label*="immagini"]',
            '[data-testid*="photo"]',
            '[data-testid*="image"]',
        ]
        for selector in hidden_source_selectors:
            try:
                elements = html_content.find(selector)
            except Exception:
                elements = []
            hidden_count = max(hidden_count, self._hidden_picture_count_from_elements(elements))

        hidden_xpaths = [
            "//main//section//section//div//figure//button//div//div",
            "//main//figure//button//div//div",
            "//figure//button//*[normalize-space()]",
            "//figure//button",
        ]
        for xpath in hidden_xpaths:
            try:
                elements = html_content.xpath(xpath)
            except Exception:
                elements = []
            hidden_count = max(hidden_count, self._hidden_picture_count_from_elements(elements))

        if hidden_count == 0:
            html_text = getattr(html_content, "html", "") or ""
            for match in re.findall(
                r">\s*(?:\+|&plus;)\s*(\d{1,3})(?:\s*(?:foto|photo|photos|immagini|images))?\s*<",
                html_text,
                flags=re.IGNORECASE,
            ):
                hidden_count = max(hidden_count, int(match))

        visible_count = len(visible_urls)
        if visible_count == 0:
            try:
                visible_count = max(
                    len(html_content.find('button[class*="item-thumbnail"]')),
                    len(html_content.find("section figure")),
                    len(html_content.find("main figure")),
                )
            except Exception:
                visible_count = 0
        if visible_count == 0 and hidden_count > 0 and urls:
            visible_count = 1
        picture_count = max(len(urls), visible_count + hidden_count if (visible_count or hidden_count) else 0)
        if picture_count == 0 and fallback_primary_url:
            picture_count = 1

        return {
            "PrimaryImageUrl": urls[0] if urls else "",
            "FullImageUrls": urls,
            "VisiblePictureCount": int(visible_count),
            "HiddenPictureCount": int(hidden_count),
            "PictureCount": int(picture_count),
        }

    def _model_metadata_path_from_best_row(self, sweep_run: Path, row: pd.Series) -> Path:
        prefix = sweep_run.name
        search_name = str(row.get("search_name"))
        approach = str(row.get("approach"))
        seed = int(float(row.get("seed", 42)))
        return Path("data/experiments/deal_finder/models") / f"{prefix}_{search_name}_{approach}_seed{seed}_metadata.json"

    def load_best_model_metadata_for_search(
        self,
        search_name: str,
        *,
        sweep_run: Path | str = DEFAULT_MODEL_SWEEP_RUN,
    ) -> dict[str, Any] | None:
        sweep_path = Path(sweep_run)
        best_path = sweep_path / "best_by_search.csv"
        if not best_path.exists():
            self.logger.warning("Deal-finder best_by_search.csv not found: %s", best_path)
            return None
        best = pd.read_csv(best_path)
        match = best[best["search_name"].astype(str).str.lower() == str(search_name).lower()]
        if match.empty:
            self.logger.info("No deal-finder model configured for search=%s", search_name)
            return None
        row = match.iloc[0]
        metadata_path = self._model_metadata_path_from_best_row(sweep_path, row)
        if not metadata_path.exists():
            self.logger.warning("Deal-finder metadata not found for search=%s: %s", search_name, metadata_path)
            return None
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["metadata_path"] = str(metadata_path)
        return metadata

    def score_live_rows_for_search(
        self,
        search_name: str,
        df: pd.DataFrame,
        *,
        sweep_run: Path | str = DEFAULT_MODEL_SWEEP_RUN,
        low_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_SCORE_THRESHOLD,
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        scored_at = datetime.now().astimezone().isoformat()
        metadata = self.load_best_model_metadata_for_search(search_name, sweep_run=sweep_run)
        if metadata is None:
            out["DealFinderScore"] = np.nan
            out["DealFinderScoreBand"] = "not_scored"
            out["DealFinderScoredAt"] = scored_at
            return out

        try:
            from experiments.deal_finder.model_sweep import add_engineered_snapshot_features, add_image_features
            from experiments.deal_finder.modeling import load_pickle, score_with_model

            work = out.copy()
            if "SearchName" not in work.columns:
                work["SearchName"] = search_name
            work = add_engineered_snapshot_features(work)
            if bool(metadata.get("requires_images")):
                work = add_image_features(work)
            for col in metadata.get("numeric_features", []) or []:
                if col not in work.columns:
                    work[col] = np.nan
            for col in metadata.get("text_features", []) or []:
                if col not in work.columns:
                    work[col] = ""

            model = load_pickle(Path(str(metadata["artifact_path"])))
            scores = score_with_model(model, work)
            out["DealFinderScore"] = np.asarray(scores, dtype=float)
            out["DealFinderScoreBand"] = "score_middle"
            out.loc[out["DealFinderScore"] <= float(low_threshold), "DealFinderScoreBand"] = "score_low"
            out.loc[out["DealFinderScore"] >= float(high_threshold), "DealFinderScoreBand"] = "score_high"
            out["DealFinderModel"] = metadata.get("approach", "")
            out["DealFinderModelSearch"] = metadata.get("search_name", search_name)
            out["DealFinderModelMetadata"] = metadata.get("metadata_path", "")
            out["DealFinderThreshold"] = float(metadata.get("threshold", np.nan))
            out["DealFinderLowThreshold"] = float(low_threshold)
            out["DealFinderHighThreshold"] = float(high_threshold)
            out["DealFinderScoredAt"] = scored_at
            return out
        except Exception as exc:
            self.logger.warning("Deal-finder scoring failed for search=%s: %s: %s", search_name, type(exc).__name__, exc)
            out["DealFinderScore"] = np.nan
            out["DealFinderScoreBand"] = "score_failed"
            out["DealFinderScoredAt"] = scored_at
            out["DealFinderScoreError"] = f"{type(exc).__name__}: {exc}"
            return out

    def select_score_extremes(
        self,
        scored_df: pd.DataFrame,
        *,
        low_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_SCORE_THRESHOLD,
    ) -> pd.DataFrame:
        if scored_df.empty or "DealFinderScore" not in scored_df.columns:
            return scored_df.iloc[0:0].copy()
        out = scored_df.copy()
        scores = pd.to_numeric(out["DealFinderScore"], errors="coerce")
        mask_low = scores <= float(low_threshold)
        mask_high = scores >= float(high_threshold)
        selected = out[mask_low | mask_high].copy()
        if selected.empty:
            return selected
        selected["FullScrapeReason"] = "score_low"
        selected.loc[pd.to_numeric(selected["DealFinderScore"], errors="coerce") >= float(high_threshold), "FullScrapeReason"] = "score_high"
        return selected.reset_index(drop=True)

    def collect_full_item_payload(
        self,
        row_dict: dict,
        *,
        search_name: str,
        reason: str,
        no_residential: bool = False,
        image_mode: str = "html",
    ) -> tuple[dict | None, dict | None]:
        scraped_at = datetime.now().astimezone().isoformat()
        data_id = row_dict.get("Dataid")
        link = row_dict.get("Link")
        if not link:
            failure = dict(row_dict)
            failure.update({
                "SearchName": search_name,
                "FullScrapeReason": reason,
                "FullScrapedAt": scraped_at,
                "FullScrapeStatus": "MissingLink",
                "FullScrapeError": "Missing Link",
            })
            return None, failure
        try:
            item_info, seller_info = self.scrape_single_product(
                url=link,
                data_id=data_id,
                get_images=True,
                image_mode=image_mode,
                no_residential=no_residential,
                fetch_sleep=0,
                fetch_max_attempts=1,
                post_fetch_sleep=0,
            )
            if not item_info and not seller_info:
                failure = dict(row_dict)
                failure.update({
                    "SearchName": search_name,
                    "FullScrapeReason": reason,
                    "FullScrapedAt": scraped_at,
                    "FullScrapeStatus": "FetchFailed",
                    "FullScrapeError": "Item page could not be loaded",
                })
                return None, failure
            out = dict(row_dict)
            out.update({
                "SearchName": search_name,
                "FullScrapeReason": reason,
                "FullScrapedAt": scraped_at,
                "FullScrapeStatus": "OK",
                "Description": item_info.get("Description", ""),
                "Condition": item_info.get("Condition", ""),
                "Upload_date": item_info.get("Upload_date", ""),
                "Interested_count": item_info.get("Interested_count", np.nan),
                "View_count": item_info.get("View_count", np.nan),
                "SellerName": seller_info.get("SellerName", item_info.get("SellerName", "")),
                "SellerId": seller_info.get("SellerId", item_info.get("SellerId", "")),
                "Location": seller_info.get("Location", ""),
                "ReviewsCount": seller_info.get("ReviewsCount", np.nan),
                "Stars": seller_info.get("Stars", np.nan),
                "PrimaryImageUrl": item_info.get("PrimaryImageUrl", ""),
                "FullImageUrls": json.dumps(item_info.get("FullImageUrls", []), ensure_ascii=True),
                "VisiblePictureCount": item_info.get("VisiblePictureCount", 0),
                "HiddenPictureCount": item_info.get("HiddenPictureCount", 0),
                "PictureCount": item_info.get("PictureCount", 0),
            })
            if item_info.get("Images"):
                out["Images"] = item_info.get("Images")
            return out, None
        except Exception as exc:
            failure = dict(row_dict)
            failure.update({
                "SearchName": search_name,
                "FullScrapeReason": reason,
                "FullScrapedAt": scraped_at,
                "FullScrapeStatus": "Error",
                "FullScrapeError": f"{type(exc).__name__}: {exc}",
            })
            return None, failure

    def collect_and_store_full_items(
        self,
        rows_df: pd.DataFrame,
        *,
        search_name: str,
        reason: str,
        max_workers: int = 2,
        no_residential: bool = False,
        image_mode: str = "html",
        skip_existing: bool = False,
        output_subdir: str | None = None,
    ) -> dict[str, Any]:
        paths = self._full_scrape_paths(search_name, output_subdir=output_subdir)
        if rows_df.empty:
            return {"requested": 0, "processed": 0, "succeeded": 0, "failed": 0, "skipped_existing": 0}

        work = rows_df.copy()
        if "FullScrapeReason" in work.columns:
            reason_by_key = {self._identity_key(row): str(row.get("FullScrapeReason") or reason) for _, row in work.iterrows()}
        else:
            reason_by_key = {}

        skipped_existing = 0
        if skip_existing:
            existing_keys = self._identity_keys_for_frame(self._read_csv_or_empty(paths["items"]))
            keep_rows = []
            for _, row in work.iterrows():
                if self._identity_key(row) in existing_keys:
                    skipped_existing += 1
                    continue
                keep_rows.append(row.to_dict())
            work = pd.DataFrame(keep_rows)

        if work.empty:
            return {
                "requested": int(len(rows_df)),
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped_existing": int(skipped_existing),
            }

        rows = [row.to_dict() for _, row in work.iterrows()]
        successes: list[dict] = []
        failures: list[dict] = []
        events: list[dict] = []

        def run_one(row: dict) -> tuple[dict | None, dict | None]:
            row_reason = reason_by_key.get(self._identity_key(row), reason)
            return self.collect_full_item_payload(
                row,
                search_name=search_name,
                reason=row_reason,
                no_residential=no_residential,
                image_mode=image_mode,
            )

        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            futures = [ex.submit(run_one, row) for row in rows]
            for fut in as_completed(futures):
                success, failure = fut.result()
                event_source = success if success is not None else failure
                if success is not None:
                    successes.append(success)
                if failure is not None:
                    failures.append(failure)
                if event_source is not None:
                    events.append({
                        "SearchName": event_source.get("SearchName", search_name),
                        "Dataid": event_source.get("Dataid", ""),
                        "Link": event_source.get("Link", ""),
                        "FullScrapeReason": event_source.get("FullScrapeReason", reason),
                        "FullScrapedAt": event_source.get("FullScrapedAt", datetime.now().astimezone().isoformat()),
                        "FullScrapeStatus": event_source.get("FullScrapeStatus", "Unknown"),
                        "DealFinderScore": event_source.get("DealFinderScore", np.nan),
                    })

        self._append_csv(paths["items"], successes, dedupe=True)
        self._append_csv(paths["failures"], failures, dedupe=False)
        self._append_csv(paths["events"], events, dedupe=False)
        return {
            "requested": int(len(rows_df)),
            "processed": int(len(rows)),
            "succeeded": int(len(successes)),
            "failed": int(len(failures)),
            "skipped_existing": int(skipped_existing),
            "items_path": str(paths["items"]),
            "failures_path": str(paths["failures"]),
            "events_path": str(paths["events"]),
        }

    def score_and_collect_extremes_for_live_rows(
        self,
        search_name: str,
        df: pd.DataFrame,
        *,
        no_residential: bool = False,
        low_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
        high_threshold: float = DEFAULT_HIGH_SCORE_THRESHOLD,
        max_workers: int = 2,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        scored = self.score_live_rows_for_search(
            search_name,
            df,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        extremes = self.select_score_extremes(
            scored,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        summary = {"extreme_rows": int(len(extremes)), "full_scrape": {}}
        if not extremes.empty:
            summary["full_scrape"] = self.collect_and_store_full_items(
                extremes,
                search_name=search_name,
                reason="score_extreme",
                max_workers=max_workers,
                no_residential=no_residential,
                image_mode="html",
                skip_existing=False,
            )
        return scored, summary

    def fetch_page_and_check(self, item, get_images=False, check_venduto=False, get_upload_date=False):
        return self.inspect_item_page(
            item,
            get_images=get_images,
            check_venduto=check_venduto,
            get_upload_date=get_upload_date,
            initial_delay=15,
        )

    def scrape_products_serial(self, dictionary, search_count, pages_to_scrape, workers, get_images=False):
        data = []
        seller_rows = []
        stats = {
            'pages_attempted': 0,
            'pages_loaded': 0,
            'products_seen': 0,
            'products_valid': 0,
            'worker_failures': 0,
        }
        webpage = self.create_webpage(dictionary)

        for page in range(pages_to_scrape):
            stats['pages_attempted'] += 1
            new_webpage = webpage + '&page=' + str(page + 1)
            self.logger.info('Full scrape search=%s page=%s url=%s', dictionary.search, page + 1, new_webpage)

            html_content = self.get_page_content(new_webpage, timeout=60, sleep=5)
            if html_content is None:
                self.logger.warning('Skipping full scrape page=%s because fetch failed', page + 1)
                continue

            stats['pages_loaded'] += 1
            products = html_content.find('.new-item-box__container')
            all_likes_counts = html_content.find('.u-background-white.u-flexbox.u-align-items-center.new-item-box__favourite-icon')
            stats['products_seen'] += len(products)

            self.logger.info('Full scrape catalog page=%s yielded %s products', page + 1, len(products))
            if len(products) == 0 and page < 10:
                break

            metas = []
            for product in products[:20]:
                meta = self.extract_catalog_item_meta(product, all_likes_counts, page, search_count, get_images=get_images)
                if meta:
                    meta['get_images'] = get_images
                    metas.append(meta)
                    stats['products_valid'] += 1

            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = [exe.submit(scrape_worker, (self, meta)) for meta in metas]
                for fut in as_completed(futures):
                    try:
                        meta, seller_data = fut.result()
                        data.append(meta)
                        seller_rows.append(seller_data)
                    except Exception as exc:
                        stats['worker_failures'] += 1
                        self.logger.warning('Full scrape worker failed on page=%s: %s', page + 1, exc)

            self.logger.info(
                'Full scrape progress search=%s page=%s rows=%s sellers=%s worker_failures=%s',
                dictionary.search,
                page + 1,
                len(data),
                len(seller_rows),
                stats['worker_failures'],
            )

        self.logger.info('Full scrape summary search=%s stats=%s', dictionary.search, stats)
        return data, seller_rows

    #scrape the specific web page of an item
    def scrape_single_product(
        self,
        url,
        data_id,
        get_images=False,
        image_mode="html",
        no_residential=False,
        allow_residential_fallback=True,
        fetch_sleep=10,
        fetch_max_attempts=3,
        post_fetch_sleep=0,
    ): #dictionary was a parameter
        
        html_content = self.get_page_content(
            url,
            sleep=fetch_sleep,
            max_attempts=fetch_max_attempts,
            allow_residential_fallback=allow_residential_fallback,
            no_residential=no_residential,
        )

        if html_content is None:
            return {}, {}

        page_exists = True              ### ???

        #f the page exists then get all the data, else remove the row from the df (for now)
        if page_exists:
            #get reviews count and star rating
            if post_fetch_sleep:
                time.sleep(post_fetch_sleep)
            try:
                reviews_element = html_content.find("div[class='web_ui__Rating__rating web_ui__Rating__small']", first=True).text
                reviews_count = int(reviews_element)
            except (ValueError, TypeError):
                reviews_count = 0  # seller has no reviews ("Nessuna recensione" or similar)
            except Exception:
                reviews_count = 0

            # try:
            try:
                stars_element = html_content.find("div[class='web_ui__Rating__rating web_ui__Rating__small']", first=True)
                stars_text = stars_element.attrs.get("aria-label").split(" ")
                if len(stars_text) == 10:
                    stars = stars_element.attrs.get("aria-label").split(" ")[6]
                else:
                    stars = stars_element.attrs.get("aria-label").split(" ")[2]
            except Exception as exc:
                self.logger.debug('Could not parse seller stars for item=%s: %s', data_id, exc)
                stars = -1
            # stars = self.driver.find_elements(By.XPATH, "//div[@class='web_ui__Rating__star web_ui__Rating__full']")

            try:
                #get location
                location_element = html_content.find("div[class='u-flexbox u-align-items-baseline']", first=True)

                # html_content.find("div.details-list__item-value--redesign[itemprop='location']", first=True)
                location = location_element.text if location_element else "Not found"            
                # location = self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='location']").text
            except Exception as exc:
                self.logger.debug('Could not parse seller location for item=%s: %s', data_id, exc)
                location = "Unknown"

            try:
                views_count_element = html_content.find('div.details-list__item-value[itemprop="view_count"] span', first=True)

                # views_count_element = html_content.find("span[class='web_ui__Text__text web_ui__Text__subtitle web_ui__Text__left web_ui__Text__bold']", first=True)
                views_count = int(views_count_element.text) if views_count_element else -1  
                # views_count = int(self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='view_count']").text)
            except Exception as exc:
                self.logger.debug('Could not parse views count for item=%s: %s', data_id, exc)
                views_count = -1
            
            
            #get interested people
            try:
                interested_count_element = html_content.find('div.details-list__item-value[itemprop="interested"] span', first=True)
                interested_count = int(interested_count_element.text.split(" ")[0]) if interested_count_element else -1  
                # interested_count = int(self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='interested']").text.split(" ")[0])
            except Exception as exc:
                self.logger.debug('Could not parse interested count for item=%s: %s', data_id, exc)
                interested_count = -1

            try:
                upload_date = html_content.find('div.details-list__item-value[itemprop="upload_date"] span', first=True).text
                # #get upload date
                # upload_date = " ".join(
                #             html_content.find('div.details-list__item-value[itemprop="upload_date"] span', first=True)
                #     # self.driver.find_element(By.XPATH, "//div[@class='details-list__item-value' and @itemprop='upload_date']").text.split()[:-1]
                #     )
            except Exception as exc:
                self.logger.debug('Could not parse upload date for item=%s: %s', data_id, exc)
                upload_date = "Unknown"
            
            try:
                #get item description
                item_description = html_content.find("span[class='web_ui__Text__text web_ui__Text__body web_ui__Text__left web_ui__Text__format']", first=True).text
                # item_description = self.driver.find_element(By.XPATH, "//span[@class='web_ui__Text__text web_ui__Text__body web_ui__Text__left web_ui__Text__format']").text
            except Exception as exc:
                self.logger.debug('Could not parse item description for item=%s: %s', data_id, exc)
                item_description = "Unknown"
            try:
                #get seller name
                seller_name = html_content.find(f"span[data-testid*='profile-username']", first=True).text
            except Exception as exc:
                self.logger.debug('Could not parse seller name for item=%s: %s', data_id, exc)
                seller_name = "Unknown"
            # except:
            #     print("problem finding something")
            #     return [], []
            
            try:
                item_condition = html_content.find('div.details-list__item-value[itemprop="status"] span', first=True).text

                # item_condition_element = html_content.find("div[data-testid='item-attributes-status']", first=True)
                # item_condition = item_condition_element.find("div[class='details-list__item-value--redesign']", first=True).text
            except Exception as exc:
                self.logger.debug('Could not parse item condition for item=%s: %s', data_id, exc)
                item_condition = "Unknown"

            # print(f"condition = {item_condition}")
            new_seller_row = {
                    "SellerId": " ",
                    "SellerName": seller_name,
                    "Location": location,
                    # "ItemCondition": item_condition,
                    # "ItemDescription": item_description,
                    "ReviewsCount": reviews_count,
                    "Stars": stars
            }

            fallback_primary = None
            try:
                fallback_primary = html_content.find('meta[property="og:image"]', first=True)
                fallback_primary = fallback_primary.attrs.get("content") if fallback_primary else None
            except Exception:
                fallback_primary = None

            image_metadata = self.extract_item_page_image_metadata(html_content, fallback_primary_url=fallback_primary)

            ### get images or not depending on what is set in the bool parameter get_images
            if get_images and str(image_mode).lower() == "selenium":
                try:
                    image_urls = self.get_all_product_images(url)
                except Exception as exc:
                    self.logger.warning('Error getting images for item=%s: %s', data_id, exc)
                    image_urls = image_metadata.get("FullImageUrls", [])
                if image_urls:
                    image_metadata["FullImageUrls"] = image_urls
                    image_metadata["PrimaryImageUrl"] = image_urls[0]
                    image_metadata["PictureCount"] = max(int(image_metadata.get("PictureCount", 0) or 0), len(image_urls))
            elif get_images:
                image_urls = image_metadata.get("FullImageUrls", [])
            else:
                image_urls = []

            page_price = self.extract_item_page_price(html_content)

            new_row = {
                "Images": image_urls,
                "Interested_count": interested_count,
                "View_count": views_count,
                "Description": item_description,
                "Condition": item_condition,
                "Upload_date": upload_date,
                "Price": page_price,
                # "Dataid": data_id,
                "SellerId": " ",
                "SellerName": seller_name,
                "PrimaryImageUrl": image_metadata.get("PrimaryImageUrl", ""),
                "FullImageUrls": image_metadata.get("FullImageUrls", []),
                "VisiblePictureCount": image_metadata.get("VisiblePictureCount", 0),
                "HiddenPictureCount": image_metadata.get("HiddenPictureCount", 0),
                "PictureCount": image_metadata.get("PictureCount", 0),
            }

        else:
            new_row = []
            print("The page doesnt exist")
        
        # print(f"new row = {new_row}")
        return new_row, new_seller_row

        # if len(stars) >= 4 and reviews_count > 3:
        #     print("almeno 4 stelle e 3 reviews")
        # else:
        #     print("non abbastanza stelle o reviews") 

    def remove_not_actually_sold_items(self, new_df, old_df, non_really_sold_items_ids):
        print("I'm removing items that are already present in old_df")
        print(f"Len before dropping duplicates = {len(new_df)}")

        if len(new_df) == 0:
            print("No new items to process.")
            return [], old_df, []

        link_list_old = list(old_df["Link"].values)
        link_list_new = list(new_df["Link"].values)

        # Remove items that are already in old_df
        for link in link_list_new:
            if link in link_list_old:
                old_df.loc[old_df["Link"] == link, "SearchDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                # if old_df:
                new_df = new_df.drop(new_df[new_df['Link'] == link].index)

        new_items_count = len(new_df)
        print(f"Len after dropping duplicates = {new_items_count}")


        # Marking the items that are in old_df but not in new_df (in those there are some fake sold and some real sold)
        for link in link_list_old:
            if link not in link_list_new:
                old_df.loc[old_df["Link"] == link, "MarketStatus"] = "Sold"

        
        print("Marked sold items")


        # Removing items that are not really sold
        items_sold_to_check = old_df.loc[old_df["MarketStatus"] == "Sold"]

        items_sold_to_check = items_sold_to_check[:20]


        if len(items_sold_to_check) > 0:
            print(f"Before removing non really sold {len(items_sold_to_check)}")
            items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(non_really_sold_items_ids)]
            print(f"After removing non really sold {len(items_sold_to_check)}")
            # items_sold_to_check = items_sold_to_check[~items_sold_to_check['Dataid'].isin(list(full_scraped_df["Dataid"].values))]
        
        # Parallel execution
        actually_sold_items = []

        max_workers = min(8, multiprocessing.cpu_count())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers based on system/network capacity
            futures = [executor.submit(self.fetch_page_and_check, row)
                    for _, row in items_sold_to_check.iterrows()]

            for future in as_completed(futures):
                item, sold, status = future.result()
                if sold:
                    actually_sold_items.append(item)
                    print(f"Item sold for real: {item['Title']} + {item['Link']}")
                else:
                    if status == "On Sale":
                        try:
                            non_really_sold_items_ids.add(int(item["Dataid"]))
                            old_df.loc[old_df["Link"] == item["Link"], "MarketStatus"] = "On Sale"
                        except (TypeError, ValueError) as exc:
                            self.logger.warning('Failed to mark non-really-sold item=%s: %s', item.get('Dataid'), exc)
                    print(f"Item not sold: {item['Title']}")

        print("Parallel scraping complete.")


        return new_df, old_df, actually_sold_items

    def remove_and_manage_old_items_in_search(self, old_df, new_df, unsold_full_scraped):

        print("I'm about to drop items in the old_df to make space for the new ones")


        new_items_count = len(new_df)

        # Make space for the new items deleting the most old ones
        min_search = old_df["SearchCount"].min()
        max_pag = old_df["Page"].max()
        if new_items_count:
            while len(old_df) + new_items_count > 900:
                print(f"Before drop: LEN = {len(old_df)}, New Items Count = {new_items_count}")
                
                print(f"Min SearchCount: {min_search}, Max Page: {max_pag}")
                rows_to_drop = old_df[(old_df["SearchCount"] == min_search) & (old_df["Page"] == max_pag)]
                
                if rows_to_drop.empty:
                    print("No rows to drop, breaking to avoid infinite loop.")
                else:        
                    unsold_full_scraped = pd.concat([unsold_full_scraped, rows_to_drop], ignore_index=True)
                    old_df = old_df.drop(rows_to_drop.index)
                    print(f"After drop: LEN = {len(old_df)}")
                
                if max_pag == 1:
                    min_search += 1
                    max_pag = 10
                else:
                    max_pag -= 1            
        
        return old_df, unsold_full_scraped
        

    def compare_and_save_df_serial(self, new_df, old_df, new_seller_df, old_seller_df, unsold_full_scraped_df, sold_full_scraped_df,  non_really_sold_items_ids):
        print("in compare and save")

        # if len(new_df) > 100:
        #     new_df = new_df[:100]

                  
        # returns updated new_df, old_df and actually_sold_items
        new_df, old_df, actually_sold_items = self.remove_not_actually_sold_items(new_df, old_df, non_really_sold_items_ids)


        if not actually_sold_items:
            print("No actually sold items found.")
        else:
            new_sold_df = pd.DataFrame(actually_sold_items)

            if "Dataid" in new_sold_df.columns:
                old_df = old_df[~old_df["Dataid"].isin(new_sold_df["Dataid"])].copy()
                sold_dedupe_subset = ["Dataid"]
            else:
                old_df = old_df[~old_df["Link"].isin(new_sold_df["Link"])].copy()
                sold_dedupe_subset = ["Link"]

            sold_full_scraped_df = pd.concat([sold_full_scraped_df, new_sold_df], ignore_index=True)
            sold_full_scraped_df = sold_full_scraped_df.drop_duplicates(subset=sold_dedupe_subset, keep="first")
            sold_full_scraped_df.to_csv(f"/home/ale/Desktop/vinted/Vinted_New_Version/data/sold_df.csv", index=False)
            print(f"Actually sold items: {len(actually_sold_items)}")

        # Remove and manage old items in the search
        old_df, unsold_full_scraped_df = self.remove_and_manage_old_items_in_search(old_df, new_df, unsold_full_scraped_df)

        full_unsold_dedupe_subset = ["Dataid"] if "Dataid" in unsold_full_scraped_df.columns else ["Link"] if "Link" in unsold_full_scraped_df.columns else None
        if full_unsold_dedupe_subset:
            unsold_full_scraped_df = unsold_full_scraped_df.drop_duplicates(subset=full_unsold_dedupe_subset, keep="first")
        unsold_full_scraped_df.to_csv(f"/home/ale/Desktop/vinted/Vinted_New_Version/data/unsold_df.csv", index=False)

        if old_seller_df["SellerId"].notna().any():
            max_id = old_seller_df["SellerId"].max()
        else:
            max_id = 0
            print("No previous seller id")

        for index, row in new_df.iterrows():  
            if row["SellerName"] in old_seller_df["SellerName"].values:
                seller_id = old_seller_df.loc[old_seller_df["SellerName"] == row["SellerName"], "SellerId"].values[0]
                new_df.at[index, "SellerId"] = seller_id  # Modify directly in DataFrame
                new_seller_df.drop(new_seller_df[new_seller_df["SellerName"] == row["SellerName"]].index, inplace=True)
            else:
                max_id += 1
                new_df.at[index, "SellerId"] = max_id  # Modify directly in DataFrame
                new_seller_df.loc[new_seller_df["SellerName"] == row["SellerName"], "SellerId"] = max_id


        old_seller_df = pd.concat([old_seller_df, new_seller_df], ignore_index=True)
        old_seller_df.to_csv(f"/home/ale/Desktop/vinted/Vinted_New_Version/data/sellers_df.csv", index=False)            

        if len(new_df) == 0:
            print("No new items to process.")
        else:
            print(old_df.index.duplicated().any())  # True if there are duplicates
            print(new_df.index.duplicated().any())  # True if there are duplicates

            old_df = pd.concat([old_df, new_df], ignore_index=True)
            old_df.to_csv(f"/home/ale/Desktop/vinted/Vinted_New_Version/data/old_df.csv", index=False)
            
        print("New seller df:")
        print(new_seller_df)


        # Save new items
        # if len(new_df) > 0:
        #     print("concateno il nuovo dataset") 

        #     print(f"len new_df = {len(new_df)}")
        #     old_df = pd.concat([old_df, new_df], ignore_index=True)
        #     old_df = old_df.drop_duplicates(subset=["Link"], keep='first', inplace=False)
        #     old_df.to_csv(f"prova/old_csv.csv", index=False)
        #     print("finito di concatenare")

            # notif.sendMessage(f"Nuova Ricerca: {input_search}, {len(new_items)} Nuovi Items")


            # send message and download main image

            # count = 0
            # for index, row in enumerate(new_df):
            #     #send whatsapp messages
            #     # notif.sendMessage(f"Item {count}: {row.iloc[0]} '  ' {row.iloc[4]}")
            #     count += 1
            #     #download images
            #     data_id = row["Dataid"]
            #     img_link = row["Image"]
            #     if(img_link != ""):
            #         utils.ensure_path_exists(f'{input_search}/{input_search} images')
            #         utils.download_image(img_link, f'{input_search}/{input_search} images/{data_id}')



        # if not removed_items.empty:
        #     for row in removed_items:
        #     last_row = utils.get_last_non_empty_row_excel(f"{input_search}/removed_items {input_search}.xlsx")
        #     with pd.ExcelWriter(f"{input_search}/removed_items {input_search}.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        #         removed_items.to_excel(writer, sheet_name='Sheet1', index=False, header=True, startrow= ++last_row)
        # else:
        #     print("nessun articolo è stato venduto")

        # Save the current state of the data
        # new_df.to_csv(file_path, index=False)

    #fill the database with additional data scraping every item's webpage



    ##fill the database with additional data scraping every item's webpage
    # def complete_df_with_sigle_scrapes(self, dictionary):

    #     csv_path = f"{dictionary['search']}/{dictionary['search']}.csv"

    #     images_root_folder = f"{dictionary['search']}/{dictionary['search']} images"
        
    #     #read csv to modify it
    #     df = pd.read_csv(csv_path)

    #     print(f"len df = {len(df)}")
    #     #initialize a list of new rows to add to the csv
    #     new_rows = []
    #     new_seller_rows = []

    #     seller_csv_path = "Sellers.csv"
    #     seller_df = pd.read_csv(seller_csv_path)
        

    #     #loop through the dataset to detect the row which are missing the info from the single product scrape
    #     for index, row in df.iterrows():  

    #         # print(f"row = {row['Images']}")
    #         #if images is empty means that the row doesn't have the complete info
    #         #also check that the folder is not created already, if it is we can skip it
    #         # if pd.isna(row["Images"]) and not os.path.exists(f"{dictionary['search']}/{dictionary['search']} images/{row['Dataid']}"):
    #         # if len(list(row["Images"])) == 0:
    #             # self.driver = self.init_driver()

    #             #get the extra data
    #             new_row, new_seller_row = self.scrape_single_product(str(row["Link"]), row["Dataid"])

    #             # if new_row:

    #             #     #maybe this can removed
    #             #     df["Images"] = df["Images"].astype("object")

    #             #     #fill the images cell with the list of images just scraped
    #             #     df.at[index, "Images"] = new_row["Images"]

    #             #     # Remove the images from the new_row becauase right above i populated the column "images" in the original df
    #             #     first_key = next(iter(new_row))
    #             #     new_row.pop(first_key)

    #             #     #download and store all the images
    #             #     image_folder_path = utils.download_all_images(df.at[index, "Images"], dictionary, new_row["Dataid"])
                    
    #             #     #check the images to recognize if the item is what we want
    #             #     is_item_right = dataset_cleaner.check_single_item_images(dictionary, image_folder_path)
    #             # else: #if new_row is empty means that the page doeas exist anymore
    #             #     is_item_right = False

    #             #if the item is correct we store it otherwise we drop the whole row in the csv
    #             new_rows.append(new_row)
    #             new_seller_rows.append(new_seller_row)
    #             # if is_item_right:
    #             #     new_rows.append(new_row)
    #             # else:
    #             #     df.drop(index, inplace=True)  # Drop the row in the main DataFrame
    #         # time.sleep(10)
    #     max_id = seller_df["SellerId"].max()

    #     columns_seller = ["SellerId", "SellerName", "Location", "ReviewsCount", "Stars"]
    #     temp_seller_df = pd.DataFrame(new_seller_rows, columns=columns_seller)
    #     temp_seller_df.drop_duplicates(subset=["SellerName"], keep='first', inplace=True)

        
    #     # seller_df.to_csv(seller_csv_path, index=False)

    #     #create a temporary df with the new rows
    #     columns = ["Interested_count", "View_count", "Item_description", "Upload_date", "Dataid", "SellerId", "SellerName"]
    #     complementary_df = pd.DataFrame(new_rows, columns=columns)

    #     for index, row in complementary_df.iterrows():  
    #         if row["SellerName"] in seller_df["SellerName"].values:
    #             seller_id = seller_df.loc[seller_df["SellerName"] == row["SellerName"], "SellerId"].values[0]
    #             complementary_df.at[index, "SellerId"] = seller_id  # Modify directly in DataFrame
    #             temp_seller_df.drop(temp_seller_df[temp_seller_df["SellerName"] == row["SellerName"]].index, inplace=True)
    #         else:
    #             max_id += 1
    #             complementary_df.at[index, "SellerId"] = max_id  # Modify directly in DataFrame
    #             temp_seller_df.loc[temp_seller_df["SellerName"] == row["SellerName"], "SellerId"] = max_id

    #     #maybe this row can be removed
    #     df.reset_index(drop=True, inplace=True)  # This removes the old index

    #     #add the temporary df with the new rows to the original df
    #     new_df = df.set_index('Dataid').combine_first(complementary_df.set_index('Dataid')).reset_index()
    #     new_df.to_csv(f"{dictionary['search']}/{dictionary['search']}.csv", index=False)
        
        
    #     new_seller_df = pd.concat([seller_df, temp_seller_df], ignore_index=True)
    #     new_seller_df.to_csv(seller_csv_path, index=False)

    #     # add all new images and info to items 
    #     # new_df = pd.merge(df, complementary_df, on="Dataid", how="left")

    #     #overwrite the csv with the updated data
