from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from utils_lib.download_tracker import estimate_response_bytes, record_download


SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")
IMAGE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.vinted.it/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def safe_token(value: object, fallback: str) -> str:
    text = SAFE_TOKEN_RE.sub("_", str(value or "").strip()).strip("._-")
    return text or fallback


def normalize_image_sources(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


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


def build_cache_base_path(*, dataset_root: Path, data_id: str, image_index: int) -> Path:
    return dataset_root / safe_token(data_id, "unknown_id") / f"image_{image_index:02d}"


def find_existing_cache_file(base_path: Path) -> Path | None:
    for candidate in sorted(base_path.parent.glob(base_path.name + ".*")):
        return candidate
    return None


def download_one_image(url: str, base_path: Path, timeout: float = 20.0, overwrite: bool = False) -> str:
    existing = find_existing_cache_file(base_path)
    if existing is not None and not overwrite:
        return str(existing.resolve())

    base_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers=IMAGE_REQUEST_HEADERS, timeout=timeout)
    body = response.content
    record_download(
        kind="listing_image",
        transport="direct",
        url=url,
        bytes_downloaded=estimate_response_bytes(response, body=body),
        status_code=response.status_code,
        ok=response.ok,
        metadata={"cache_path": str(base_path.parent.resolve())},
    )
    response.raise_for_status()
    suffix = infer_extension(url, response)
    final_path = base_path.with_suffix(suffix)
    final_path.write_bytes(body)
    return str(final_path.resolve())


def download_listing_images_to_cache(
    image_urls: Any,
    *,
    dataset_root: Path,
    data_id: object,
    timeout: float = 20.0,
    overwrite: bool = False,
) -> list[str]:
    urls = normalize_image_sources(image_urls)
    data_id_text = str(data_id or "unknown")
    local_paths: list[str] = []

    for index, url in enumerate(urls, start=1):
        if not url:
            continue
        base_path = build_cache_base_path(
            dataset_root=dataset_root,
            data_id=data_id_text,
            image_index=index,
        )
        try:
            local_paths.append(
                download_one_image(
                    url,
                    base_path,
                    timeout=timeout,
                    overwrite=overwrite,
                )
            )
        except Exception:
            continue

    return local_paths
