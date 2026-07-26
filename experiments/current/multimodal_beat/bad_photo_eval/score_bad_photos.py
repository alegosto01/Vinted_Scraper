"""Phases 2-6: build the single scored bad-photo table.

Loads the alignment-locked CSV + frozen CLIP embeddings, computes:
  v1_generic_clip_score  (frozen baseline: zero_shot_badimg.py logic)
  v2_typed_clip_score    (frozen baseline: zero_shot_typed_badimg.py logic)
  v3 generic CLIP defect margins (clip_margins.py)     -> clip_overall_margin etc.
  simple_technical first-image features (simple_features.py)
Globally deduplicates by item_id (aggregating search names) and writes one row per
unique listing to data/scored_bad_photo_candidates.csv.

No combined/final score is computed here (that waits for manual labels).

Usage:
  python score_bad_photos.py --out data/scored_bad_photo_candidates.csv
  python score_bad_photos.py --max-items 100 --out data/smoke_scored.csv   # smoke
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MB = HERE.parent
sys.path.insert(0, str(MB))
sys.path.insert(0, str(HERE))

from alignment import load_aligned, normalize_ids, EMBEDDING_MODEL  # noqa: E402
import clip_margins  # noqa: E402
import simple_features as sf  # noqa: E402
# frozen baselines
from zero_shot_badimg import GOOD as V1_GOOD, BAD as V1_BAD  # noqa: E402
from zero_shot_typed_badimg import TYPES as V2_TYPES, classify as v2_classify  # noqa: E402

SCORING_VERSION = "bad_photo_eval.v1"
EMB_CACHE_ID = "img_clip_live19k"


def _clip_text(prompts):
    """Encode + L2-normalize CLIP text prompts; return (matrix, logit_scale)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(EMBEDDING_MODEL).eval()
    p = CLIPProcessor.from_pretrained(EMBEDDING_MODEL)
    with torch.no_grad():
        inp = p(text=prompts, return_tensors="pt", padding=True)
        t = m.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        v = m.text_projection(t.pooler_output).numpy()
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return v.astype(np.float32), float(m.logit_scale.exp())


def v1_generic_clip(img: np.ndarray) -> np.ndarray:
    """Frozen v1: softmax over 5 good + 5 bad prompts, sum P(bad)."""
    txt, scale = _clip_text(V1_GOOD + V1_BAD)
    sims = img @ txt.T * scale
    e = np.exp(sims - sims.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    return prob[:, len(V1_GOOD):].sum(1).astype(np.float32)


def v2_typed_clip(img: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Frozen v2: type via search+title (image-CLIP fallback), then within-type
    sigmoid(bad-good). Returns (bad_score, item_type)."""
    names = list(V2_TYPES)
    itype = np.array([v2_classify(s, t) for s, t in
                      zip(df.get("SearchName", ""), df.get("Title", ""))], dtype=object)
    need = np.where(pd.isna(itype))[0]
    if len(need):
        det, _ = _clip_text([V2_TYPES[t]["detect"][0] for t in names])
        guess = np.array(names)[(img[need] @ det.T).argmax(1)]
        for j, i in enumerate(need):
            itype[i] = guess[j]
    gb, owner, kind = [], [], []
    for t in names:
        gb += [V2_TYPES[t]["good"][0], V2_TYPES[t]["bad"][0]]
        owner += [t, t]; kind += ["good", "bad"]
    gbmat, scale = _clip_text(gb)
    owner = np.array(owner); kind = np.array(kind)
    gbsim = img @ gbmat.T * scale
    bad = np.zeros(len(img), dtype=np.float32)
    for t in names:
        rows = np.where(itype == t)[0]
        if not len(rows):
            continue
        cg = np.where((owner == t) & (kind == "good"))[0]
        cb = np.where((owner == t) & (kind == "bad"))[0]
        lg = gbsim[np.ix_(rows, cg)].mean(1)
        lb = gbsim[np.ix_(rows, cb)].mean(1)
        bad[rows] = 1.0 / (1.0 + np.exp(lg - lb))
    return bad, itype.astype(str)


def deduplicate(df: pd.DataFrame, item_ids: np.ndarray, mask: np.ndarray,
                id_index: dict) -> pd.DataFrame:
    """Phase 2: one row per unique item_id. Aggregate search_names; keep the first
    (lowest original row index) occurrence as the representative for scores/image."""
    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["_item_id"] = item_ids
    work["_search"] = work.get("SearchName", pd.Series([""] * len(work))).astype(str)
    grp = work.groupby("_item_id", sort=False)
    rep = grp["_row"].min()                      # representative row = first occurrence
    search_names = grp["_search"].apply(lambda s: sorted(set(s)))
    all_rows = grp["_row"].apply(list)
    uniq = pd.DataFrame({"item_id": rep.index, "rep_row": rep.values})
    uniq["search_names"] = uniq["item_id"].map(search_names)
    uniq["primary_search"] = work.set_index("_item_id")["_search"].groupby(level=0).first().reindex(uniq["item_id"]).values
    uniq["title"] = work.set_index("_item_id").get("Title").groupby(level=0).first().reindex(uniq["item_id"]).values if "Title" in work else ""
    uniq["original_row_indices"] = uniq["item_id"].map(all_rows)
    # image path + availability by item_id (order-independent, same for all dup rows)
    paths, avail = [], []
    for iid, r in zip(uniq["item_id"], uniq["rep_row"]):
        f = sf.resolve_first_image(iid, id_index)
        paths.append(str(f) if f else "")
        avail.append(int(mask[r]))
    uniq["image_path"] = paths
    uniq["image_available"] = avail
    return uniq.sort_values("item_id").reset_index(drop=True)


def build_table(max_items: int | None, out: Path):
    df, emb, mask, item_ids, manifest = load_aligned()
    print(f"loaded aligned: {len(df)} rows, {int(mask.sum())} valid images")

    # ---- per-row CLIP scores (all frozen; identical across duplicate rows) ----
    print("v1 generic clip ...")
    v1 = v1_generic_clip(emb)
    print("v2 typed clip ...")
    v2, itype = v2_typed_clip(emb, df)
    print("v3 generic defect margins ...")
    margins = clip_margins.score_margins(emb)

    per_row = pd.DataFrame({"item_id": item_ids,
                            "v1_generic_clip_score": v1,
                            "v2_typed_clip_score": v2,
                            "v2_item_type": itype})
    for k, arr in margins.items():
        per_row[k] = arr

    # ---- Phase 2 dedup ----
    id_index = sf.build_id_index()
    uniq = deduplicate(df, item_ids, mask, id_index)
    print(f"unique items after dedup: {len(uniq)}")

    # attach representative-row CLIP scores to unique table
    rep_scores = per_row.iloc[uniq["rep_row"].values].reset_index(drop=True)
    rep_scores = rep_scores.drop(columns=["item_id"])
    table = pd.concat([uniq.reset_index(drop=True), rep_scores], axis=1)

    if max_items:
        table = table.head(max_items).copy()
        print(f"[smoke] limited to {len(table)} items")

    # ---- Phase 4 simple features on unique first images ----
    print("simple features (first image per unique item) ...")
    feats = []
    n = len(table)
    for i, p in enumerate(table["image_path"].tolist()):
        feats.append(sf.extract_features(p) if p else sf.extract_features("__missing__"))
        if (i + 1) % 500 == 0:
            print(f"  features {i+1}/{n}", flush=True)
    fdf = pd.DataFrame(feats)
    table = pd.concat([table.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)
    table = sf.percentile_scores(table)
    from blur_score import add_blur_score        # texture-normalized blur ranker (validated)
    table = add_blur_score(table)

    # ---- provenance ----
    table["source_csv_sha256"] = manifest["source_csv_sha256"]
    table["embedding_cache_id"] = EMB_CACHE_ID
    table["scoring_version"] = SCORING_VERSION
    table["search_names"] = table["search_names"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
    table["original_row_indices"] = table["original_row_indices"].apply(
        lambda x: ",".join(map(str, x)) if isinstance(x, list) else x)

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nWROTE {out}  ({len(table)} rows, {table.shape[1]} cols)")
    print("cols:", list(table.columns))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "scored_bad_photo_candidates.csv"))
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args()
    build_table(a.max_items, Path(a.out))
