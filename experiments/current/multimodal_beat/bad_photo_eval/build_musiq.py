"""Score every listing's first image with MUSIQ (pretrained no-reference IQA).

MUSIQ beat the hand-crafted blur_score on human labels (grouped-CV AUROC 0.84 vs 0.78
for blur, 0.84 vs 0.73 for the general bad-photo target; P@10 = 1.0 on both). It is a
pretrained model — zero training on our labels — so no overfitting. Higher MUSIQ = better
photo; we store `musiq_bad = -musiq` so higher = worse, matching the other rankers.

One-time image pass (~19k). Resumable: writes progress to a cache npy keyed by row.

Usage:
  python build_musiq.py --scores data/scored_with_blur.csv --out data/scored_with_musiq.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CACHE = HERE / "data" / "musiq_all.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scored_with_blur.csv")
    ap.add_argument("--out", default="data/scored_with_musiq.csv")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import pyiqa
    from embed_blocks import build_id_index, first_image
    df = pd.read_csv(a.scores, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)
    idx = build_id_index()
    metric = pyiqa.create_metric("musiq", device=a.device)

    scores = np.load(CACHE) if CACHE.exists() and len(np.load(CACHE)) == len(df) \
        else np.full(len(df), np.nan)
    n = len(df)
    for i, iid in enumerate(df["item_id"].tolist()):
        if not np.isnan(scores[i]):
            continue
        d = idx.get(iid)
        f = first_image(d) if d else None
        if f is not None:
            try:
                scores[i] = float(metric(str(f)))
            except Exception:
                scores[i] = np.nan
        if (i + 1) % 250 == 0:
            np.save(CACHE, scores)
            print(f"  musiq {i+1}/{n} (done={int((~np.isnan(scores)).sum())})", flush=True)
    np.save(CACHE, scores)
    df["musiq"] = scores
    df["musiq_bad"] = -scores            # higher = worse, matches other rankers
    df.to_csv(a.out, index=False)
    print(f"WROTE {a.out} | musiq nonnull={int((~np.isnan(scores)).sum())}/{n}")


if __name__ == "__main__":
    main()
