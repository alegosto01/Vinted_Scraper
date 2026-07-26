"""Phase 1: recover and lock the embedding<->CSV alignment.

The cached CLIP arrays (img_clip_live19k.npy / _mask.npy) are aligned to the *row
order* of the original 20,450-row CSV, but that CSV path was never recorded. This
module proves which CSV matches and writes an item_id sidecar + manifest so every
downstream script can assert alignment cheaply.

Alignment oracle (prompt-independent):
  The mask was produced by embed_blocks.py, which walked the CSV's item_id column in
  order and set mask[i]=1 iff an image existed for csv.item_id[i]. Therefore, for the
  correctly-ordered CSV, mask[i] must equal (image resolvable for csv.item_id[i]) for
  ALL rows. A wrongly-ordered CSV fails this on a large fraction of rows.

We also cross-check content against the retained v1 review (cache/badimg_live19k/
worst.csv): its 40 item_ids must all be present in the candidate CSV.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MB = HERE.parent                       # experiments/current/multimodal_beat
CACHE = MB / "cache"
sys.path.insert(0, str(MB))
from embed_blocks import build_id_index, first_image  # noqa: E402

EMB_PATH = CACHE / "img_clip_live19k.npy"
MASK_PATH = CACHE / "img_clip_live19k_mask.npy"
ITEM_IDS_PATH = CACHE / "img_clip_live19k_item_ids.npy"
MANIFEST_PATH = CACHE / "img_clip_live19k_manifest.json"
EMBEDDING_MODEL = "openai/clip-vit-base-patch32"


def normalize_ids(series: pd.Series) -> np.ndarray:
    """str, strip whitespace, drop a single trailing '.0' from float-cast ints."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.to_numpy()


def pick_id_column(df: pd.DataFrame) -> str:
    for c in ("item_id", "Dataid"):
        if c in df.columns:
            return c
    raise ValueError(f"no item_id/Dataid column in CSV; columns={list(df.columns)[:12]}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_alignment(df: pd.DataFrame, mask: np.ndarray, id_col: str,
                     tolerance: float = 0.005):
    """Return (ok, report_dict). ok=True iff mask matches image existence within
    `tolerance` fraction of rows (0.5% default absorbs images deleted/added since
    embedding time)."""
    ids = normalize_ids(df[id_col])
    idx = build_id_index()
    exists = np.array([1 if (idx.get(i) and first_image(idx[i])) else 0 for i in ids],
                      dtype=np.int64)
    agree = (exists == mask.astype(np.int64))
    disagree = int((~agree).sum())
    frac_disagree = disagree / len(mask)
    ok = frac_disagree <= tolerance
    return ok, {
        "rows": int(len(df)),
        "mask_valid": int(mask.sum()),
        "images_now_resolvable": int(exists.sum()),
        "positions_disagree": disagree,
        "fraction_disagree": round(frac_disagree, 5),
        "tolerance": tolerance,
        "aligned": bool(ok),
    }


def content_check(df: pd.DataFrame, id_col: str) -> dict:
    """Cross-check: all 40 item_ids from the retained v1 review must be in the CSV."""
    worst = CACHE / "badimg_live19k" / "worst.csv"
    if not worst.exists():
        return {"retained_review": "missing", "checked": 0}
    w = pd.read_csv(worst)
    want = set(normalize_ids(w["item_id"]))
    have = set(normalize_ids(df[id_col]))
    missing = sorted(want - have)
    return {"retained_review_ids": len(want),
            "present_in_csv": len(want) - len(missing),
            "missing_ids": missing[:10]}


def recover(csv_path: Path, write: bool = True, tolerance: float = 0.005):
    emb = np.load(EMB_PATH)
    mask = np.load(MASK_PATH)
    df = pd.read_csv(csv_path, low_memory=False)
    if len(df) != len(emb):
        raise SystemExit(
            f"ROW COUNT MISMATCH: CSV {len(df)} rows vs embeddings {len(emb)}. "
            f"Wrong CSV: {csv_path}")
    id_col = pick_id_column(df)
    ok, rep = verify_alignment(df, mask, id_col, tolerance)
    content = content_check(df, id_col)
    print(f"CSV: {csv_path}")
    print("alignment:", json.dumps(rep, indent=2))
    print("content  :", json.dumps(content, indent=2))
    if not ok:
        raise SystemExit(
            f"ALIGNMENT FAILED: {rep['positions_disagree']} / {rep['rows']} positions "
            f"disagree ({rep['fraction_disagree']:.2%} > {tolerance:.2%}). "
            f"Do not run the experiment on this CSV.")
    if content.get("missing_ids"):
        raise SystemExit(
            f"CONTENT CHECK FAILED: retained review item_ids missing from CSV: "
            f"{content['missing_ids']}")
    if not write:
        return df, id_col, rep
    ids = normalize_ids(df[id_col])
    np.save(ITEM_IDS_PATH, ids.astype("U32"))
    manifest = {
        "source_csv": str(csv_path),
        "source_csv_sha256": sha256_file(csv_path),
        "row_count": int(len(df)),
        "embedding_shape": list(emb.shape),
        "mask_shape": list(mask.shape),
        "valid_image_count": int(mask.sum()),
        "embedding_model": EMBEDDING_MODEL,
        "normalized_embeddings": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "item_id_column": id_col,
        "alignment_report": rep,
        "content_check": content,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nWROTE {ITEM_IDS_PATH.name} and {MANIFEST_PATH.name}")
    return df, id_col, rep


def load_aligned():
    """Load (df, embeddings, mask, item_ids) with all alignment asserts. Used by every
    downstream scoring script."""
    if not MANIFEST_PATH.exists():
        raise SystemExit("run alignment.py first: no manifest. Alignment not locked.")
    man = json.loads(MANIFEST_PATH.read_text())
    df = pd.read_csv(man["source_csv"], low_memory=False)
    emb = np.load(EMB_PATH)
    mask = np.load(MASK_PATH)
    item_ids = np.load(ITEM_IDS_PATH, allow_pickle=False).astype(str)
    id_col = man["item_id_column"]
    assert len(df) == len(emb), f"{len(df)} != {len(emb)}"
    assert len(df) == len(mask), f"{len(df)} != {len(mask)}"
    assert len(df) == len(item_ids), f"{len(df)} != {len(item_ids)}"
    assert emb.shape[1] == 512, emb.shape
    csv_ids = normalize_ids(df[id_col])
    assert csv_ids.tolist() == item_ids.tolist(), "CSV item_id order != sidecar"
    return df, emb, mask, item_ids, man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    recover(Path(a.input_csv), write=not a.no_write, tolerance=a.tolerance)
