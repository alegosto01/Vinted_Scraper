#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import requests

from analysis_pipeline.scoring.visual_rerank import IMAGE_REQUEST_HEADERS, normalize_image_sources


SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download listing images into a local cache for stable offline analysis.")
    ap.add_argument("--input", required=True, help="Path to a CSV with Dataid and Images columns")
    ap.add_argument("--search_name", default=None, help="Search name used in the cache folder layout")
    ap.add_argument("--source_name", default=None, help="Logical dataset name inside the cache layout")
    ap.add_argument("--cache_root", default=None, help="Override the cache root directory")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional row limit for smaller tests")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def safe_token(value: object, fallback: str) -> str:
    text = SAFE_TOKEN_RE.sub("_", str(value or "").strip()).strip("._-")
    return text or fallback


def infer_extension(url: str, response: requests.Response) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return suffix
    content_type = (response.headers.get("content-type") or "").lower()
    if "webp" in content_type:
        return ".webp"
    if "png" in content_type:
        return ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "gif" in content_type:
        return ".gif"
    return ".img"


def download_one(url: str, destination: Path, timeout: float, overwrite: bool) -> tuple[bool, str]:
    if destination.exists() and not overwrite:
        return True, "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, headers=IMAGE_REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        return False, str(exc)
    suffix = infer_extension(url, response)
    final_path = destination.with_suffix(suffix)
    if final_path.exists() and not overwrite:
        return True, "cached"
    final_path.write_bytes(response.content)
    return True, str(final_path)


def build_cache_paths(
    *,
    dataset_root: Path,
    data_id: str,
    image_index: int,
) -> Path:
    return dataset_root / safe_token(data_id, "unknown_id") / f"image_{image_index:02d}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    search_name = args.search_name or input_path.parent.name
    source_name = args.source_name or input_path.stem
    cache_root = Path(args.cache_root) if args.cache_root else ROOT / "data" / "simple_scrape" / safe_token(search_name, "unknown_search") / "image_cache"
    out_dir = cache_root / safe_token(source_name, "dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    local_image_lists: list[list[str]] = []
    primary_paths: list[str] = []
    status_rows: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        data_id = str(getattr(row, "Dataid", "unknown"))
        image_urls = normalize_image_sources(getattr(row, "Images", None))
        local_paths: list[str] = []
        for idx, url in enumerate(image_urls, start=1):
            base_path = build_cache_paths(
                dataset_root=out_dir,
                data_id=data_id,
                image_index=idx,
            )
            ok, detail = download_one(url, base_path, args.timeout, args.overwrite)
            local_path = detail if ok and detail not in {"cached"} else ""
            if ok and detail == "cached":
                for candidate in sorted(base_path.parent.glob(base_path.name + ".*")):
                    local_path = str(candidate.resolve())
                    break
            if local_path:
                local_paths.append(local_path)
            status_rows.append(
                {
                    "Dataid": data_id,
                    "ImageIndex": idx,
                    "ImageUrl": url,
                    "Downloaded": bool(ok),
                    "LocalImagePath": local_path,
                    "Detail": detail,
                }
            )
        local_image_lists.append(local_paths)
        primary_paths.append(local_paths[0] if local_paths else "")

    df = df.copy()
    df["LocalImagePaths"] = [json.dumps(paths) for paths in local_image_lists]
    df["LocalPrimaryImagePath"] = primary_paths

    manifest_path = out_dir / "image_manifest.csv"
    cached_csv_path = out_dir / f"{input_path.stem}_with_local_images.csv"
    summary_path = out_dir / "image_cache_summary.json"

    pd.DataFrame(status_rows).to_csv(manifest_path, index=False)
    df.to_csv(cached_csv_path, index=False)

    downloaded_count = sum(1 for row in status_rows if row["Downloaded"])
    summary = {
        "input_csv": str(input_path),
        "search_name": search_name,
        "source_name": source_name,
        "cache_root": str(cache_root),
        "n_rows": int(len(df)),
        "n_image_attempts": int(len(status_rows)),
        "n_downloaded": int(downloaded_count),
        "n_failed": int(len(status_rows) - downloaded_count),
        "manifest_csv": str(manifest_path),
        "cached_csv": str(cached_csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
