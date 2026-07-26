"""Rank Vinted search results by multilingual-title and first-image similarity.

This prototype is deliberately independent from the bad-photo detector. Similarities
are cosine scores, not probabilities; no match threshold is fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import logging
import math
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("vinted_product_matcher")
OUTPUT_COLUMNS = [
    "target_item_id", "target_title", "candidate_item_id", "candidate_title",
    "listing_url", "search_name", "title_similarity", "image_similarity",
    "combined_score", "title_rank", "image_rank", "combined_rank", "image_status",
]


def normalize_title(value: object) -> str:
    """Conservative normalization: retain punctuation, numbers, and model codes."""
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text.strip().lower())


def prepare_title_text(title: str, role: str, model_name: str) -> str:
    if "e5" not in model_name.lower():
        return title
    return ("query: " if role == "query" else "passage: ") + title


def validate_weights(title_weight: float, image_weight: float) -> None:
    if title_weight < 0 or image_weight < 0:
        raise ValueError("weights must be non-negative")
    if not math.isclose(title_weight + image_weight, 1.0, abs_tol=1e-9):
        raise ValueError("title_weight and image_weight must sum to 1")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("model returned a zero embedding")
    return values / norms


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def _cache_path(
    cache_dir: Path, kind: str, model_name: str, item_id: str, content_hash: str
) -> Path:
    item_key = _content_hash(str(item_id))
    return cache_dir / kind / _slug(model_name) / f"{item_key}_{content_hash}.npy"


def _load_embedding(path: Path) -> np.ndarray | None:
    try:
        value = np.load(path, allow_pickle=False).astype(np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            return None
        norm = np.linalg.norm(value)
        return value / norm if norm else None
    except (OSError, ValueError):
        return None


def _save_embedding(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as tmp:
        temporary = Path(tmp.name)
        np.save(tmp, np.asarray(value, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


class E5TitleEncoder:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode(self, titles: Sequence[str], role: str) -> np.ndarray:
        tokens = self.tokenizer(
            [prepare_title_text(title, role, self.model_name) for title in titles],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return _unit_rows(pooled.float().cpu().numpy())


class DINOImageEncoder:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
        return _unit_rows(pooled.float().cpu().numpy())


class ImageResolver:
    def __init__(self, download_dir: Path):
        self.download_dir = download_dir

    def resolve(self, source: object) -> tuple[Path | None, str]:
        if source is None or pd.isna(source) or not str(source).strip():
            return None, "missing"
        value = str(source).strip()
        try:
            if value.startswith(("http://", "https://")):
                suffix = Path(value.split("?", 1)[0]).suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    suffix = ".img"
                path = self.download_dir / f"{_content_hash(value)}{suffix}"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    request = urllib.request.Request(
                        value, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(request, timeout=30) as response:
                        payload = response.read()
                    path.write_bytes(payload)
            else:
                path = Path(value).expanduser().resolve()
                if not path.is_file():
                    return None, "missing"
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            return path, "ok"
        except Exception as exc:
            LOG.warning("Unreadable image %s: %s", value, exc)
            return None, "unreadable"


def _embed_titles(
    item_ids: Sequence[str],
    titles: Sequence[str],
    role: str,
    encoder,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
) -> tuple[list[np.ndarray], int]:
    result: list[np.ndarray | None] = [None] * len(titles)
    missing: list[tuple[int, Path]] = []
    for index, (item_id, title) in enumerate(zip(item_ids, titles)):
        path = _cache_path(
            cache_dir, f"titles_{role}", model_name, item_id, _content_hash(title)
        )
        cached = _load_embedding(path) if path.exists() else None
        if cached is None:
            missing.append((index, path))
        else:
            result[index] = cached
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        vectors = encoder.encode([titles[index] for index, _ in batch], role)
        for (index, path), vector in zip(batch, vectors):
            result[index] = vector
            _save_embedding(path, vector)
    return [value for value in result if value is not None], len(missing)


def _embed_images(
    item_ids: Sequence[str],
    sources: Sequence[object],
    encoder,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    resolver: ImageResolver,
) -> tuple[list[np.ndarray | None], list[str], list[Path | None], int]:
    result: list[np.ndarray | None] = [None] * len(sources)
    statuses: list[str] = []
    paths: list[Path | None] = []
    missing: list[tuple[int, Path, Path]] = []
    for index, (item_id, source) in enumerate(zip(item_ids, sources)):
        image_path, status = resolver.resolve(source)
        paths.append(image_path)
        statuses.append(status)
        if image_path is None:
            continue
        cache_path = _cache_path(
            cache_dir, "images", model_name, item_id, _file_hash(image_path)
        )
        cached = _load_embedding(cache_path) if cache_path.exists() else None
        if cached is None:
            missing.append((index, cache_path, image_path))
        else:
            result[index] = cached
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        try:
            vectors = encoder.encode([image_path for _, _, image_path in batch])
            encoded = list(zip(batch, vectors))
        except Exception as exc:
            LOG.warning("Image batch failed (%s); retrying individually", exc)
            encoded = []
            for entry in batch:
                try:
                    encoded.append((entry, encoder.encode([entry[2]])[0]))
                except Exception as item_exc:
                    LOG.warning("Image embedding failed for %s: %s", entry[2], item_exc)
                    statuses[entry[0]] = "embedding_error"
        for (index, cache_path, _), vector in encoded:
            vector = _unit_rows(np.asarray(vector)[None, :])[0]
            result[index] = vector
            _save_embedding(cache_path, vector)
    return result, statuses, paths, len(missing)


def deterministic_ranks(values: Sequence[float], item_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = [index for index, value in enumerate(values) if np.isfinite(value)]
    valid.sort(key=lambda index: (-values[index], str(item_ids[index]), index))
    ranks = np.full(len(values), np.nan)
    for rank, index in enumerate(valid, 1):
        ranks[index] = rank
    return ranks


@dataclass
class MatchConfig:
    target_item_id: str
    target_title: str
    target_image: str
    candidates: Path
    out_dir: Path
    title_model: str = "intfloat/multilingual-e5-base"
    image_model: str = "facebook/dinov2-base"
    title_weight: float = 0.5
    image_weight: float = 0.5
    top_k: int = 50
    device: str = "auto"
    batch_size: int = 32
    rank_by: str = "combined"
    candidate_id_col: str = "Dataid"
    title_col: str = "Title"
    image_col: str = "Images"
    listing_url_col: str = "Link"
    search_name_col: str = "SearchName"
    cache_dir: Path | None = None


def _cosines(target: np.ndarray | None, candidates: Sequence[np.ndarray | None]) -> np.ndarray:
    if target is None:
        return np.full(len(candidates), np.nan)
    return np.asarray([
        float(target @ candidate) if candidate is not None else np.nan
        for candidate in candidates
    ])


def _display_uri(path: Path | None, original: object) -> str:
    if path is not None:
        return path.resolve().as_uri()
    value = "" if original is None or pd.isna(original) else str(original)
    return value if value.startswith(("http://", "https://")) else ""


def _write_gallery(
    path: Path,
    ranked: pd.DataFrame,
    rank_column: str,
    top_k: int,
    target_title: str,
    target_uri: str,
    candidate_uris: dict[int, str],
) -> None:
    cards = []
    for index, row in ranked.head(max(top_k, 0)).iterrows():
        item_id = str(row["candidate_item_id"])
        scores = (
            f"title={row['title_similarity']:.4f} · "
            f"image={row['image_similarity']:.4f} · "
            f"combined={row['combined_score']:.4f}"
        )
        cards.append(f"""
        <article>
          <img src="{html.escape(candidate_uris.get(index, ''))}" loading="lazy">
          <div><b>#{html.escape(str(row[rank_column]))}</b> {html.escape(str(row['candidate_title']))}</div>
          <code>{html.escape(scores)}</code>
          <div><a href="{html.escape(str(row['listing_url']))}" target="_blank" rel="noopener">Vinted listing</a></div>
          <select class="label" data-id="{html.escape(item_id)}">
            <option value="">unlabelled</option>
            <option>same_product</option><option>different_product</option><option>uncertain</option>
          </select>
        </article>""")
    document = f"""<!doctype html>
<meta charset="utf-8"><title>Vinted product-match review</title>
<style>
body{{font:14px system-ui;margin:20px}} .target{{display:flex;gap:16px;align-items:center}}
.target img,article img{{width:180px;height:180px;object-fit:contain;background:#eee}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}}
article{{border:1px solid #ccc;padding:10px}} article>*{{margin:5px 0;display:block}}
</style>
<h1>Target: {html.escape(target_title)}</h1>
<div class="target"><img src="{html.escape(target_uri)}"><span>Ranked by {rank_column}</span></div>
<p><button id="download">Download labels CSV</button></p><main class="grid">{''.join(cards)}</main>
<script>
const key="vinted_match_labels"; const labels=JSON.parse(localStorage.getItem(key)||"{{}}");
document.querySelectorAll(".label").forEach(s=>{{s.value=labels[s.dataset.id]||"";
s.onchange=()=>{{labels[s.dataset.id]=s.value;localStorage.setItem(key,JSON.stringify(labels));}}}});
document.querySelector("#download").onclick=()=>{{
 let csv="candidate_item_id,manual_label\\n";
 for(const [id,label] of Object.entries(labels)) csv+=JSON.stringify(id)+","+JSON.stringify(label)+"\\n";
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));
 a.download="manual_labels.csv";a.click();URL.revokeObjectURL(a.href);
}};
</script>"""
    path.write_text(document, encoding="utf-8")


def run_matching(config: MatchConfig, title_encoder=None, image_encoder=None) -> pd.DataFrame:
    validate_weights(config.title_weight, config.image_weight)
    if config.top_k < 0 or config.batch_size < 1:
        raise ValueError("top_k must be >= 0 and batch_size must be >= 1")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.cache_dir or config.out_dir / "embedding_cache"
    candidates = pd.read_csv(config.candidates, dtype={config.candidate_id_col: str})
    required = [config.candidate_id_col, config.title_col, config.image_col]
    absent = [column for column in required if column not in candidates]
    if absent:
        raise ValueError(f"missing candidate columns: {', '.join(absent)}")
    target_id = str(config.target_item_id)
    candidates = candidates[
        candidates[config.candidate_id_col].fillna("").astype(str) != target_id
    ].copy()
    ids = candidates[config.candidate_id_col].fillna("").astype(str).tolist()
    titles = candidates[config.title_col].map(normalize_title).tolist()
    target_title = normalize_title(config.target_title)
    device = choose_device(config.device)
    LOG.info("Rows=%d device=%s", len(candidates), device)

    title_encoder = title_encoder or E5TitleEncoder(config.title_model, device)
    target_titles, target_title_misses = _embed_titles(
        [str(config.target_item_id)], [target_title], "query", title_encoder,
        config.title_model, cache_dir, config.batch_size,
    )
    candidate_titles, candidate_title_misses = _embed_titles(
        ids, titles, "passage", title_encoder, config.title_model, cache_dir,
        config.batch_size,
    )
    title_similarity = _cosines(target_titles[0], candidate_titles)

    image_encoder = image_encoder or DINOImageEncoder(config.image_model, device)
    resolver = ImageResolver(config.out_dir / "downloaded_images")
    target_images, target_status, target_paths, target_image_misses = _embed_images(
        [str(config.target_item_id)], [config.target_image], image_encoder,
        config.image_model, cache_dir, config.batch_size, resolver,
    )
    candidate_images, image_status, candidate_paths, candidate_image_misses = _embed_images(
        ids, candidates[config.image_col].tolist(), image_encoder, config.image_model,
        cache_dir, config.batch_size, resolver,
    )
    image_similarity = _cosines(target_images[0], candidate_images)
    combined = (
        config.title_weight * title_similarity
        + config.image_weight * image_similarity
    )
    def optional(column: str) -> pd.Series:
        return (
            candidates[column].fillna("")
            if column in candidates else pd.Series([""] * len(candidates))
        )
    output = pd.DataFrame({
        "target_item_id": str(config.target_item_id),
        "target_title": config.target_title,
        "candidate_item_id": ids,
        "candidate_title": candidates[config.title_col].fillna(""),
        "listing_url": optional(config.listing_url_col),
        "search_name": optional(config.search_name_col),
        "title_similarity": title_similarity,
        "image_similarity": image_similarity,
        "combined_score": combined,
        "title_rank": deterministic_ranks(title_similarity, ids),
        "image_rank": deterministic_ranks(image_similarity, ids),
        "combined_rank": deterministic_ranks(combined, ids),
        "image_status": image_status,
    })
    rank_column = f"{config.rank_by}_rank"
    output["_source_index"] = np.arange(len(output))
    output = output.sort_values(
        [rank_column, "candidate_item_id", "_source_index"],
        ascending=[True, True, True], na_position="last", kind="stable",
    )
    source_indices = output["_source_index"].astype(int).tolist()
    output = output.drop(columns="_source_index")
    csv_path = config.out_dir / "ranked_matches.csv"
    output.to_csv(csv_path, index=False, columns=OUTPUT_COLUMNS)
    candidate_uris = {
        output_index: _display_uri(candidate_paths[source_index], candidates.iloc[source_index][config.image_col])
        for output_index, source_index in zip(output.index, source_indices)
    }
    _write_gallery(
        config.out_dir / "review_gallery.html", output, rank_column, config.top_k,
        config.target_title, _display_uri(target_paths[0], config.target_image),
        candidate_uris,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "title_weight * title_similarity + image_weight * image_similarity",
        "title_weight": config.title_weight,
        "image_weight": config.image_weight,
        "rank_by": config.rank_by,
        "rows": len(output),
        "target_image_status": target_status[0],
        "models": {
            "title": {"name": config.title_model, "revision": getattr(title_encoder, "revision", None)},
            "image": {"name": config.image_model, "revision": getattr(image_encoder, "revision", None)},
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "pandas", "Pillow")
        },
        "cache": {
            "root": str(cache_dir.resolve()),
            "title_embeddings_created": target_title_misses + candidate_title_misses,
            "image_embeddings_created": target_image_misses + candidate_image_misses,
        },
        "outputs": {
            "ranked_csv": str(csv_path.resolve()),
            "review_gallery": str((config.out_dir / "review_gallery.html").resolve()),
        },
    }
    (config.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOG.info("Wrote %s and review_gallery.html", csv_path)
    return output


def parse_args() -> MatchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-item-id", required=True)
    parser.add_argument("--target-title", required=True)
    parser.add_argument("--target-image", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--title-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--image-model", default="facebook/dinov2-base")
    parser.add_argument("--title-weight", type=float, default=0.5)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rank-by", choices=("title", "image", "combined"), default="combined")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--candidate-id-col", default="Dataid")
    parser.add_argument("--title-col", default="Title")
    parser.add_argument("--image-col", default="Images")
    parser.add_argument("--listing-url-col", default="Link")
    parser.add_argument("--search-name-col", default="SearchName")
    args = parser.parse_args()
    return MatchConfig(**vars(args))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_matching(parse_args())


if __name__ == "__main__":
    main()
