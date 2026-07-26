"""Build a compact kept/non-kept HTML from matcher output and audit labels."""
from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path

import pandas as pd


def validate_conditions(rows: pd.DataFrame, expected: str) -> None:
    wrong = rows[rows["Condition"].fillna("").str.casefold() != expected.casefold()]
    if not wrong.empty:
        raise ValueError(
            f"{len(wrong)} candidates do not match target condition {expected!r}"
        )


def image_uri(url: object, download_dir: Path) -> str:
    value = "" if pd.isna(url) else str(url)
    local = Path(value).expanduser()
    if local.is_file():
        return local.resolve().as_uri()
    key = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    cached = next(iter(download_dir.glob(f"{key}.*")), None)
    return cached.resolve().as_uri() if cached else value


def card(row: pd.Series, download_dir: Path) -> str:
    decision = str(row["decision"])
    return f"""<article class="{html.escape(decision)}">
<img src="{html.escape(image_uri(row['Images'], download_dir))}" loading="lazy">
<h3>#{int(row['combined_rank'])} · {html.escape(str(row['candidate_title']))}</h3>
<p><b>€{float(row['Price']):.2f}</b> · {html.escape(str(row['reason']))}</p>
<small>{html.escape(str(row['Condition']))}</small>
<code>title {row['title_similarity']:.3f} · image {row['image_similarity']:.3f} · combined {row['combined_score']:.3f}</code>
<p><a href="{html.escape(str(row['listing_url']))}" target="_blank" rel="noopener">Open Vinted</a></p>
</article>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--target-title", required=True)
    parser.add_argument("--target-image", required=True, type=Path)
    parser.add_argument("--target-price", required=True, type=float)
    parser.add_argument("--target-musiq", required=True, type=float)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--source-note", default="")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    ranked = pd.read_csv(args.ranked, dtype={"candidate_item_id": str})
    candidates = pd.read_csv(args.candidates, dtype={"Dataid": str})
    labels = pd.read_csv(args.labels, dtype={"candidate_item_id": str})
    rows = ranked.merge(
        candidates[["Dataid", "Price", "Images", "Condition"]],
        left_on="candidate_item_id", right_on="Dataid", how="left",
    ).merge(labels, on="candidate_item_id", how="left")
    rows["decision"] = rows["decision"].fillna("uncertain")
    rows["reason"] = rows["reason"].fillna("Insufficient evidence for exact product model")
    validate_conditions(rows, args.condition)
    kept = rows[rows["decision"] == "kept"].copy()
    rejected = rows[rows["decision"] != "kept"].copy()
    median = kept["Price"].median()
    median_text = (
        f"median asking price €{median:.2f}"
        if pd.notna(median) else "no reliable comparable median"
    )
    download_dir = args.ranked.parent / "downloaded_images"
    document = f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(args.target_title)} match audit</title>
<style>
body{{font:14px system-ui;margin:24px;background:#f6f6f6;color:#222}}
.target{{display:flex;gap:18px;align-items:center;background:white;padding:16px}}
.target img,article img{{width:180px;height:180px;object-fit:contain;background:#eee}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
article{{background:white;border:1px solid #ccc;padding:10px}} article.different_variant{{border-color:#d66}}
article.uncertain{{opacity:.78}} h2{{margin-top:34px}} code{{font-size:12px}}
.note{{max-width:950px;padding:12px;background:#fff5cc}}
</style>
<h1>Product-match audit: {html.escape(args.target_title)}</h1>
<div class="target"><img src="{args.target_image.resolve().as_uri()}"><div>
<b>Target price €{args.target_price:.2f}</b><br>MUSIQ bad-photo score: {args.target_musiq:.2f}<br>
Condition: {html.escape(args.condition)}<br>
{len(kept)} conservative exact matches · {median_text}</div></div>
<p class="note">E5 + DINOv2 generated ranking. Human audit created kept/non-kept split because no
threshold has been fitted. Price is displayed after matching and was not used in scores.
“Uncertain” remains non-kept for conservative valuation.
{html.escape(args.source_note)}</p>
<h2>Kept — same product and condition ({len(kept)})</h2>
<div class="grid">{''.join(card(row, download_dir) for _, row in kept.iterrows())}</div>
<h2>Non-kept — different variant or uncertain exact model ({len(rejected)})</h2>
<div class="grid">{''.join(card(row, download_dir) for _, row in rejected.iterrows())}</div>
"""
    args.out.write_text(document, encoding="utf-8")
    print(
        f"wrote={args.out} kept={len(kept)} non_kept={len(rejected)} "
        f"kept_median_price={median_text}"
    )


if __name__ == "__main__":
    main()
