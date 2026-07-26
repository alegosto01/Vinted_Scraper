"""Phases 7-8 of the bad-photo detector spec: build a blind evaluation set
and HTML gallery from a scored-candidates CSV.

Reads data/scored_bad_photo_candidates.csv (one row per item_id), selects a
stratified sample per search x method (v1/v2/v3/simple/random), dedups by
item_id, injects a fraction of deliberate repeats for labeling-consistency
checks, and emits:

  - eval_manifest.json          config + counts
  - eval_candidates_private.csv full info (private key, do not share with labeler)
  - blind_label_sheet.csv       blind_id + image ref + empty label columns
  - blind_gallery.html          self-contained review gallery (images only)

Everything is driven by a single seeded RNG so re-running with the same
--seed reproduces identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEARCHES = [
    "prada",
    "gucci",
    "nike",
    "ps4",
    "griffati_donna_all",
    "griffati_uomo_all",
]

METHOD_COLUMNS = {
    "v1": "v1_generic_clip_score",
    "v2": "v2_typed_clip_score",
    "v3": "clip_overall_margin",
    "simple": "simple_overall_score",
}

BUCKETS = list(METHOD_COLUMNS) + ["random"]

PRIVATE_COLUMNS = [
    "blind_id",
    "item_id",
    "search_names",
    "primary_search",
    "title",
    "image_path",
    "is_repeat",
    "selection_reasons",
    "v1_generic_clip_score",
    "v2_typed_clip_score",
    "clip_overall_margin",
    "simple_overall_score",
    "clip_top_defect",
    "simple_top_defect",
]

LABEL_COLUMNS = [
    "technical_quality",
    "hurts_listing_presentation",
    "fixable_by_retake",
    "defect_tags",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", default="data/scored_bad_photo_candidates.csv")
    p.add_argument("--per-method-search", type=int, default=8)
    p.add_argument("--random-per-search", type=int, default=8)
    p.add_argument("--repeat-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=24072026)
    p.add_argument("--out-dir", default="data/evaluation")
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional smoke cap on the candidate pool size (for testing).",
    )
    p.add_argument(
        "--searches",
        nargs="*",
        default=None,
        help="Override the target search list (defaults to the 6 real searches).",
    )
    return p.parse_args()


def row_matches_search(search_names: str, search: str) -> bool:
    if not isinstance(search_names, str) or not search_names:
        return False
    return search in search_names.split("|")


def build_pool(df: pd.DataFrame) -> pd.DataFrame:
    pool = df[(df["image_available"] == 1) & df["image_path"].notna() & (df["image_path"] != "")]
    return pool.reset_index(drop=True)


def select_search_buckets(
    pool: pd.DataFrame,
    search: str,
    per_method: int,
    per_random: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Return {bucket_name: dataframe subset} for one search."""
    mask = pool["search_names"].apply(lambda s: row_matches_search(s, search))
    sub = pool[mask]
    if sub.empty:
        return {}

    buckets: dict[str, pd.DataFrame] = {}
    top_region_ids: set = set()
    margin_k = max(per_method * 3, 50)

    for method, col in METHOD_COLUMNS.items():
        ranked = sub.dropna(subset=[col]).sort_values(col, ascending=False)
        buckets[method] = ranked.head(per_method)
        top_region_ids.update(ranked.head(margin_k)["item_id"].tolist())

    remaining = sub[~sub["item_id"].isin(top_region_ids)]
    if len(remaining) > per_random:
        idx = rng.choice(len(remaining), size=per_random, replace=False)
        buckets["random"] = remaining.iloc[np.sort(idx)]
    else:
        buckets["random"] = remaining

    return buckets


def build_selection(
    df: pd.DataFrame,
    searches: list[str],
    per_method: int,
    per_random: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pool = build_pool(df)
    reasons: dict[int, list[str]] = {}
    rows_by_item: dict[int, pd.Series] = {}

    for search in searches:
        buckets = select_search_buckets(pool, search, per_method, per_random, rng)
        for bucket_name, bucket_df in buckets.items():
            reason = f"{search}:{bucket_name}"
            for _, row in bucket_df.iterrows():
                item_id = row["item_id"]
                reasons.setdefault(item_id, []).append(reason)
                rows_by_item.setdefault(item_id, row)

    if not rows_by_item:
        return pd.DataFrame(columns=list(df.columns) + ["selection_reasons"])

    # Stable order: item_id ascending (any deterministic order is fine, the
    # final gallery order is re-shuffled with the seed later anyway).
    ordered_ids = sorted(rows_by_item.keys())
    records = []
    for item_id in ordered_ids:
        row = rows_by_item[item_id].copy()
        row["selection_reasons"] = ";".join(reasons[item_id])
        records.append(row)

    return pd.DataFrame(records).reset_index(drop=True)


def add_repeats(
    unique_df: pd.DataFrame, repeat_fraction: float, rng: np.random.Generator
) -> pd.DataFrame:
    unique_df = unique_df.copy()
    unique_df["is_repeat"] = 0

    n_unique = len(unique_df)
    n_repeats = math.ceil(repeat_fraction * n_unique) if n_unique else 0
    n_repeats = min(n_repeats, n_unique)

    if n_repeats == 0:
        return unique_df

    repeat_idx = rng.choice(n_unique, size=n_repeats, replace=False)
    repeat_idx = np.sort(repeat_idx)
    repeats = unique_df.iloc[repeat_idx].copy()
    repeats["is_repeat"] = 1

    return pd.concat([unique_df, repeats], ignore_index=True)


def assign_blind_ids_and_shuffle(
    full_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    full_df = full_df.reset_index(drop=True)
    n = len(full_df)
    width = max(4, len(str(n)))
    full_df["blind_id"] = [f"B{i + 1:0{width}d}" for i in range(n)]

    order = rng.permutation(n)
    return full_df.iloc[order].reset_index(drop=True)


def sha256_of_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    searches: list[str],
    n_unique: int,
    n_repeats: int,
    n_total: int,
    source_sha: str | None,
) -> None:
    manifest = {
        "seed": args.seed,
        "per_method_search": args.per_method_search,
        "random_per_search": args.random_per_search,
        "repeat_fraction": args.repeat_fraction,
        "searches": searches,
        "source_csv": str(args.scores),
        "source_csv_sha256": source_sha,
        "n_unique_items": n_unique,
        "n_repeats": n_repeats,
        "n_total_rows": n_total,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "eval_manifest.json").write_text(json.dumps(manifest, indent=2))


def build_gallery_html(rows: list[dict], out_path: Path) -> None:
    label_schema = {
        "technical_quality": ["good", "bad", "uncertain", "not_item_photo"],
        "hurts_listing_presentation": ["yes", "no", "uncertain"],
        "fixable_by_retake": ["yes", "no", "uncertain"],
        "defect_tags": [
            "blur",
            "dark",
            "overexposed",
            "glare",
            "bad_crop",
            "extreme_tilt",
            "low_resolution",
            "noise",
            "clutter",
            "item_not_clear",
            "other",
        ],
    }
    rows_json = json.dumps(rows)
    schema_json = json.dumps(label_schema)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Blind Photo Quality Review</title>
<style>
  body {{ font-family: sans-serif; margin: 1.5rem; }}
  .instructions {{ background: #f4f4f4; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; }}
  .card {{ border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; max-width: 640px; }}
  .card img {{ max-width: 100%; max-height: 480px; display: block; margin-bottom: 0.75rem; }}
  .blind-id {{ font-weight: bold; margin-bottom: 0.5rem; }}
  fieldset {{ border: 1px solid #ddd; margin-bottom: 0.5rem; }}
  legend {{ font-weight: 600; }}
  label {{ margin-right: 0.75rem; }}
  textarea {{ width: 100%; }}
  #downloadBtn {{ font-size: 1rem; padding: 0.5rem 1rem; margin-bottom: 1.5rem; }}
</style>
</head>
<body>
<div class="instructions">
  <h2>Blind Photo Quality Review</h2>
  <p>Judge <strong>only the technical quality of the first/primary image</strong>
  shown below (blur, exposure, glare, cropping, resolution, clutter, tilt, etc).</p>
  <p>Do <strong>not</strong> judge the product's value, price, brand, or desirability &mdash;
  only whether the photo itself is a good or bad technical photo of an item.</p>
  <p>When done, click "Download CSV" to export your labels.</p>
</div>
<button id="downloadBtn">Download CSV</button>
<div id="gallery"></div>

<script>
const ROWS = {rows_json};
const SCHEMA = {schema_json};

function fieldName(blindId, field) {{
  return blindId + "__" + field;
}}

function buildRadioGroup(blindId, field, options) {{
  let html = `<fieldset><legend>${{field}}</legend>`;
  for (const opt of options) {{
    const id = fieldName(blindId, field) + "__" + opt;
    html += `<label><input type="radio" name="${{fieldName(blindId, field)}}" id="${{id}}" value="${{opt}}"> ${{opt}}</label>`;
  }}
  html += `</fieldset>`;
  return html;
}}

function buildCheckboxGroup(blindId, field, options) {{
  let html = `<fieldset><legend>${{field}}</legend>`;
  for (const opt of options) {{
    const id = fieldName(blindId, field) + "__" + opt;
    html += `<label><input type="checkbox" name="${{fieldName(blindId, field)}}" value="${{opt}}" id="${{id}}"> ${{opt}}</label>`;
  }}
  html += `</fieldset>`;
  return html;
}}

function renderGallery() {{
  const gallery = document.getElementById("gallery");
  let html = "";
  for (const row of ROWS) {{
    html += `<div class="card" data-blind-id="${{row.blind_id}}">`;
    html += `<div class="blind-id">${{row.blind_id}}</div>`;
    html += `<img src="${{row.image_ref}}" loading="lazy">`;
    html += buildRadioGroup(row.blind_id, "technical_quality", SCHEMA.technical_quality);
    html += buildRadioGroup(row.blind_id, "hurts_listing_presentation", SCHEMA.hurts_listing_presentation);
    html += buildRadioGroup(row.blind_id, "fixable_by_retake", SCHEMA.fixable_by_retake);
    html += buildCheckboxGroup(row.blind_id, "defect_tags", SCHEMA.defect_tags);
    html += `<fieldset><legend>notes</legend><textarea id="${{fieldName(row.blind_id, "notes")}}" rows="2"></textarea></fieldset>`;
    html += `</div>`;
  }}
  gallery.innerHTML = html;
}}

function csvEscape(value) {{
  const s = String(value ?? "");
  if (/[",\\n]/.test(s)) {{
    return '"' + s.replace(/"/g, '""') + '"';
  }}
  return s;
}}

function collectLabels() {{
  const lines = ["blind_id,image_path,technical_quality,hurts_listing_presentation,fixable_by_retake,defect_tags,notes"];
  for (const row of ROWS) {{
    const blindId = row.blind_id;
    const tq = document.querySelector(`input[name="${{fieldName(blindId, "technical_quality")}}"]:checked`);
    const hurts = document.querySelector(`input[name="${{fieldName(blindId, "hurts_listing_presentation")}}"]:checked`);
    const fixable = document.querySelector(`input[name="${{fieldName(blindId, "fixable_by_retake")}}"]:checked`);
    const tags = Array.from(document.querySelectorAll(`input[name="${{fieldName(blindId, "defect_tags")}}"]:checked`)).map(el => el.value);
    const notes = document.getElementById(fieldName(blindId, "notes")).value;
    const fields = [
      blindId,
      row.image_ref,
      tq ? tq.value : "",
      hurts ? hurts.value : "",
      fixable ? fixable.value : "",
      tags.join(";"),
      notes,
    ];
    lines.push(fields.map(csvEscape).join(","));
  }}
  return lines.join("\\n");
}}

document.getElementById("downloadBtn").addEventListener("click", () => {{
  const csv = collectLabels();
  const blob = new Blob([csv], {{ type: "text/csv" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "blind_label_sheet.csv";
  a.click();
  URL.revokeObjectURL(url);
}});

renderGallery();
</script>
</body>
</html>
"""
    out_path.write_text(html)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    scores_path = Path(args.scores)
    df = pd.read_csv(scores_path)

    if args.max_items is not None:
        df = df.head(args.max_items).reset_index(drop=True)

    searches = args.searches if args.searches else DEFAULT_SEARCHES
    present_searches = [
        s for s in searches
        if df["search_names"].apply(lambda x: row_matches_search(x, s)).any()
    ]
    if not present_searches:
        raise SystemExit(
            f"None of the target searches {searches} are present in {scores_path}"
        )

    unique_df = build_selection(
        df, present_searches, args.per_method_search, args.random_per_search, rng
    )
    n_unique = len(unique_df)

    full_df = add_repeats(unique_df, args.repeat_fraction, rng)
    n_repeats = int((full_df["is_repeat"] == 1).sum())

    full_df = assign_blind_ids_and_shuffle(full_df, rng)
    n_total = len(full_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in PRIVATE_COLUMNS:
        if col not in full_df.columns:
            full_df[col] = None
    private_df = full_df[PRIVATE_COLUMNS]
    private_df.to_csv(out_dir / "eval_candidates_private.csv", index=False)

    label_df = full_df[["blind_id", "image_path"]].copy()
    for col in LABEL_COLUMNS:
        label_df[col] = ""
    label_df.to_csv(out_dir / "blind_label_sheet.csv", index=False)

    gallery_rows = [
        {"blind_id": r["blind_id"], "image_ref": f"file://{r['image_path']}"}
        for r in full_df.to_dict(orient="records")
    ]
    build_gallery_html(gallery_rows, out_dir / "blind_gallery.html")

    source_sha = None
    if "source_csv_sha256" in df.columns and not df["source_csv_sha256"].empty:
        candidates = df["source_csv_sha256"].dropna().unique().tolist()
        if len(candidates) == 1:
            source_sha = candidates[0]
        elif candidates:
            source_sha = candidates[0]
    write_manifest(out_dir, args, present_searches, n_unique, n_repeats, n_total, source_sha)

    print(
        f"unique={n_unique} repeats={n_repeats} total={n_total} "
        f"searches={present_searches} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
