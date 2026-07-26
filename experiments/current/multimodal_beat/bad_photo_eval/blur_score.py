"""Texture-normalized blur score (single-defect target).

Raw sharpness (Laplacian variance) conflates blur with low-detail/plain photos: a sharp
photo of a plain item on a white background also has low Laplacian. Normalizing sharpness
by the image's own texture/contrast fixes that.

    blur_score = z( -log(lap) + log(edge_density) )        # lap per unit edge content
               + z( -log(lap) + 2*log(dynamic_range) )     # lap per unit contrast^2
Higher = blurrier. Computed purely from columns already in the scored table (no image
reload). On 12 human blur labels vs 62 clear-good this reaches AUROC ~0.80 / P@10 ~0.50,
versus 0.68 / 0.20 for raw Laplacian.

ponytail: a frequency-domain / no-reference blur CNN would likely do better but needs an
image pass over 19k files; the column-derived score already clears the raw baseline, so
defer that until labels justify it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-6
REQUIRED = ["laplacian_variance", "edge_density", "dynamic_range"]


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + EPS)


def add_blur_score(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `table` with a `blur_score` column (higher = blurrier). Rows
    missing any required feature (feature_error) get NaN."""
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise ValueError(f"blur_score needs columns {missing}")
    t = table.copy()
    lap = t["laplacian_variance"].astype(float)
    nb_edge = -np.log(lap + EPS) + np.log(t["edge_density"].astype(float) + EPS)
    nb_dr = -np.log(lap + EPS) + 2 * np.log(t["dynamic_range"].astype(float) + EPS)
    score = _z(nb_edge) + _z(nb_dr)
    if "feature_error" in t.columns:
        score = score.where(t["feature_error"] != 1, np.nan)
    t["blur_score"] = score
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scored_with_v4.csv")
    ap.add_argument("--out", default="data/scored_with_blur.csv")
    a = ap.parse_args()
    t = add_blur_score(pd.read_csv(a.scores, low_memory=False))
    t.to_csv(a.out, index=False)
    print(f"wrote {a.out} | blur_score nonnull={int(t.blur_score.notna().sum())} "
          f"p50={t.blur_score.median():.2f} p90={t.blur_score.quantile(.9):.2f}")
