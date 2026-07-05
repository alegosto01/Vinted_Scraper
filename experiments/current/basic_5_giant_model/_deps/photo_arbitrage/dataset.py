from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.current.basic_5_giant_model._deps.photo_arbitrage.paths import SIMPLE_SCRAPE_DIR, utc_now_iso


SEARCH_EXCLUDE_NAMES = {
    "logs",
    "tuning_reports",
    "__pycache__",
}

CANDIDATE_COLUMNS = [
    "item_id",
    "SearchName",
    "Title",
    "Brand",
    "Price",
    "Size",
    "Likes",
    "Page",
    "SearchDate",
    "MarketStatus",
    "Dataid",
    "Link",
    "Images",
    "ImageUrls",
    "LocalPrimaryImagePath",
    "LocalImagePaths",
    "PictureCount",
    "VisiblePictureCount",
    "HiddenPictureCount",
    "Description",
    "Condition",
    "Upload_date",
    "Interested_count",
    "View_count",
    "SellerName",
    "Location",
    "ReviewsCount",
    "Stars",
    "CandidateBuiltAt",
]


def normalize_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_sources(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def identity_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    if "Dataid" in df.columns:
        data_ids = df["Dataid"].map(normalize_id)
    else:
        data_ids = pd.Series([""] * len(df), index=df.index)
    if "Link" in df.columns:
        links = df["Link"].fillna("").astype(str).str.strip()
        data_ids = data_ids.where(data_ids.astype(str).str.len() > 0, links)
    return data_ids.fillna("").astype(str)


def dedupe_candidates(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["_identity"] = identity_series(out)
    out = out[out["_identity"].astype(str).str.len() > 0].copy()
    if out.empty:
        return out.drop(columns=["_identity"], errors="ignore")
    out = out.drop_duplicates("_identity", keep=keep)
    return out.drop(columns=["_identity"], errors="ignore").reset_index(drop=True)


def discover_search_dirs(root: Path = SIMPLE_SCRAPE_DIR) -> list[Path]:
    if not root.exists():
        return []
    dirs = []
    for path in root.iterdir():
        if not path.is_dir() or path.name in SEARCH_EXCLUDE_NAMES:
            continue
        if (path / "big_raw.csv").exists():
            dirs.append(path)
    return sorted(dirs, key=lambda p: p.name.lower())


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def find_cached_image_paths(search_dir: Path, data_id: Any) -> list[str]:
    data_id_text = normalize_id(data_id)
    if not data_id_text:
        return []
    item_dir = search_dir / "image_cache" / data_id_text
    if not item_dir.exists():
        return []
    paths = []
    for candidate in sorted(item_dir.glob("image_*.*")):
        if candidate.is_file():
            paths.append(str(candidate.resolve()))
    return paths


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "[]"}:
            return value
    return ""


def numeric_max(*values: Any) -> float:
    nums = []
    for value in values:
        try:
            num = float(value)
        except Exception:
            continue
        if np.isfinite(num):
            nums.append(num)
    return float(max(nums)) if nums else np.nan


def merge_full_enrichment(base: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    if base.empty or full.empty:
        return base.copy()
    left = base.copy()
    right = full.copy()
    left["_identity"] = identity_series(left)
    right["_identity"] = identity_series(right)
    right = right[right["_identity"].astype(str).str.len() > 0].drop_duplicates("_identity", keep="last")
    merge_cols = [
        "_identity",
        "Description",
        "Condition",
        "Upload_date",
        "Interested_count",
        "View_count",
        "SellerName",
        "Location",
        "ReviewsCount",
        "Stars",
        "PrimaryImageUrl",
        "FullImageUrls",
        "PictureCount",
        "VisiblePictureCount",
        "HiddenPictureCount",
    ]
    merge_cols = [col for col in merge_cols if col in right.columns]
    merged = left.merge(right[merge_cols], how="left", on="_identity", suffixes=("", "_full"))
    return merged.drop(columns=["_identity"], errors="ignore")


def build_search_candidates(search_dir: Path, *, limit: int | None = None) -> pd.DataFrame:
    raw = load_csv_or_empty(search_dir / "big_raw.csv")
    if raw.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    raw = dedupe_candidates(raw, keep="last")
    if limit is not None:
        raw = raw.head(int(limit)).copy()
    raw["SearchName"] = search_dir.name
    full = load_csv_or_empty(search_dir / "full_scrape" / "items_enriched.csv")
    merged = merge_full_enrichment(raw, full)
    built_at = utc_now_iso()
    rows = []
    for _, row in merged.iterrows():
        local_paths = normalize_sources(first_nonempty(row.get("LocalImagePaths"), ""))
        if not local_paths:
            local_paths = find_cached_image_paths(search_dir, row.get("Dataid"))
        primary_path = first_nonempty(row.get("LocalPrimaryImagePath"), local_paths[0] if local_paths else "")
        image_urls = []
        for value in (
            row.get("FullImageUrls"),
            row.get("PrimaryImageUrl"),
            row.get("Images"),
        ):
            for source in normalize_sources(value):
                if source not in image_urls:
                    image_urls.append(source)
        visible_hidden_total = numeric_max(row.get("VisiblePictureCount"), 0) + numeric_max(row.get("HiddenPictureCount"), 0)
        picture_count = numeric_max(row.get("PictureCount"), visible_hidden_total, len(local_paths), len(image_urls))
        item_id = normalize_id(row.get("Dataid")) or str(row.get("Link") or "").strip()
        out = {
            "item_id": item_id,
            "SearchName": search_dir.name,
            "Title": row.get("Title", ""),
            "Brand": row.get("Brand", ""),
            "Price": row.get("Price", np.nan),
            "Size": row.get("Size", ""),
            "Likes": row.get("Likes", np.nan),
            "Page": row.get("Page", np.nan),
            "SearchDate": row.get("SearchDate", ""),
            "MarketStatus": row.get("MarketStatus", ""),
            "Dataid": row.get("Dataid", ""),
            "Link": row.get("Link", ""),
            "Images": row.get("Images", ""),
            "ImageUrls": json.dumps(image_urls, ensure_ascii=True),
            "LocalPrimaryImagePath": str(primary_path or ""),
            "LocalImagePaths": json.dumps(local_paths, ensure_ascii=True),
            "PictureCount": picture_count,
            "VisiblePictureCount": row.get("VisiblePictureCount", np.nan),
            "HiddenPictureCount": row.get("HiddenPictureCount", np.nan),
            "Description": row.get("Description", ""),
            "Condition": row.get("Condition", ""),
            "Upload_date": row.get("Upload_date", ""),
            "Interested_count": row.get("Interested_count", np.nan),
            "View_count": row.get("View_count", np.nan),
            "SellerName": row.get("SellerName", ""),
            "Location": row.get("Location", ""),
            "ReviewsCount": row.get("ReviewsCount", np.nan),
            "Stars": row.get("Stars", np.nan),
            "CandidateBuiltAt": built_at,
        }
        rows.append(out)
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def build_candidate_dataset(
    *,
    all_searches: bool = False,
    searches: list[str] | None = None,
    limit_per_search: int | None = None,
    simple_root: Path = SIMPLE_SCRAPE_DIR,
) -> pd.DataFrame:
    if all_searches:
        search_dirs = discover_search_dirs(simple_root)
    else:
        search_dirs = [simple_root / name for name in (searches or [])]
        search_dirs = [path for path in search_dirs if (path / "big_raw.csv").exists()]
    frames = [build_search_candidates(path, limit=limit_per_search) for path in search_dirs]
    if not frames:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return pd.concat(frames, ignore_index=True)

