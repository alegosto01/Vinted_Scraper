"""v4 hybrid detector: route each defect to the signal that can actually see it.

Design (from the review of the question/filter idea):
  - PIXEL defects  : blur, underexposure, overexposure, resolution/noise
      CLIP's frozen 512-d global vector lost the fine detail these need, so they are
      scored from cheap pixel features (already in the scored table), NOT from CLIP.
  - CLIP defects   : crop, clutter, item_visibility
      genuinely semantic -> paired antonym CLIP margins.
  - HYBRID defects : glare, tilt
      pixel proxy AND CLIP margin (kept as separate candidate gates; calibration
      decides which survives).

Each CLIP question uses MATCHED (good, bad) antonym pairs. Per pair:
    pair_margin = cos(img, bad) - cos(img, good)
question margin = mean(pair_margins); disagreement = std(pair_margins).
No softmax / sigmoid / temperature / summed-prob. v3 (clip_margins.py / prompts.py)
is left frozen; this module is additive.

A "gate" is one calibrated defect signal. The hybrid score is the MAX normalized
threshold-exceedance over the gates that PASSED screening (see calibrate_gates.py):
    excess_g = (raw_g - threshold_g) / robust_scale_g
    hybrid_score = max(excess_g);  hybrid_reason = argmax_g
Max (not mean): one serious defect is enough to make a photo bad.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "clip_text_cache"
EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# ---- CLIP questions as matched (good, bad) antonym pairs (generic, no product type) ----
QUESTION_PAIRS: dict[str, list[tuple[str, str]]] = {
    "item_visibility": [
        ("the main item is clearly visible and easy to inspect",
         "the main item is obscured or difficult to inspect"),
        ("the important details of the main item are easy to see",
         "important details of the main item are hard to see"),
        ("the main item can be clearly distinguished in the photograph",
         "the main item is difficult to distinguish in the photograph"),
    ],
    "crop": [
        ("the complete main item is inside the image frame",
         "important parts of the main item are cut off by the image border"),
        ("the entire main item is visible in the photograph",
         "the main item extends outside the photograph"),
        ("the photograph includes all important parts of the main item",
         "the photograph is badly cropped around the main item"),
    ],
    "clutter": [
        ("the main item stands out clearly from a simple background",
         "the main item is difficult to distinguish from a cluttered background"),
        ("the background does not distract from the main item",
         "surrounding objects distract attention from the main item"),
        ("the main item is visually separated from its surroundings",
         "the main item is visually lost among background clutter"),
    ],
    "tilt": [
        ("the photograph is upright with a natural orientation",
         "the photograph is sideways or strongly tilted"),
        ("the main item is shown at a normal viewing angle",
         "the image has an unnatural extreme rotation"),
        ("the photograph is not strongly rotated",
         "the photograph is badly rotated and difficult to view"),
    ],
    "glare": [
        ("details of the item are visible without strong reflections",
         "strong glare or reflections hide details of the item"),
        ("the item surface can be inspected without obstructing glare",
         "bright reflections obstruct important parts of the item"),
        ("reflections do not hide important parts of the item",
         "the item is difficult to inspect because of glare"),
    ],
}
CLIP_QUESTIONS = list(QUESTION_PAIRS)

# ---- pixel-defect gates: defect -> (feature column, higher_is_worse) candidates ----
# Columns come from the scored table (simple_features.py). Multiple candidate features
# per defect; calibrate_gates screens each. `worse` marks the direction of "more bad".
PIXEL_GATES: dict[str, list[tuple[str, str]]] = {
    # defect: list of (feature_column, direction) ; direction in {"low_bad","high_bad"}
    "blur": [("laplacian_variance", "low_bad"), ("tenengrad", "low_bad"),
             ("gradient_sharpness", "low_bad"), ("edge_density", "low_bad")],
    "underexposure": [("dark_pixel_fraction", "high_bad"), ("mean_luminance", "low_bad")],
    "overexposure": [("bright_pixel_fraction", "high_bad")],
    "resolution": [("minimum_dimension", "low_bad"), ("pixel_count", "low_bad")],
    "glare_pixel": [("high_brightness_low_saturation_fraction", "high_bad"),
                    ("largest_highlight_region_fraction", "high_bad")],
}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def encode_question_pairs(cache: bool = True):
    """Return {question: (good_emb (k,512), bad_emb (k,512))}, L2-normalized, cached by
    prompt-content hash. good_emb[j]/bad_emb[j] are the j-th matched pair."""
    key = _hash(QUESTION_PAIRS)
    path = CACHE / f"question_pairs_{key}.npz"
    if cache and path.exists():
        z = np.load(path)
        return {q: (z[f"{q}__good"], z[f"{q}__bad"]) for q in CLIP_QUESTIONS}
    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(EMBEDDING_MODEL).eval()
    p = CLIPProcessor.from_pretrained(EMBEDDING_MODEL)

    def embed(texts):
        with torch.no_grad():
            inp = p(text=texts, return_tensors="pt", padding=True)
            t = m.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            v = m.text_projection(t.pooler_output).numpy()
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    out, flat = {}, {}
    for q in CLIP_QUESTIONS:
        goods = [g for g, _ in QUESTION_PAIRS[q]]
        bads = [b for _, b in QUESTION_PAIRS[q]]
        ge, be = embed(goods), embed(bads)
        out[q] = (ge, be)
        flat[f"{q}__good"], flat[f"{q}__bad"] = ge, be
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **flat)
    return out


def question_margins(img: np.ndarray, pairs=None) -> dict[str, np.ndarray]:
    """img: (N,512) L2-normalized. Returns for each question:
      q_<name>_margin       = mean_j (cos(img,bad_j) - cos(img,good_j))
      q_<name>_disagreement = std_j  (cos(img,bad_j) - cos(img,good_j))
    Higher margin = more likely that defect. Paired, so prompt count cannot compete."""
    if pairs is None:
        pairs = encode_question_pairs()
    cols: dict[str, np.ndarray] = {}
    for q, (ge, be) in pairs.items():
        pair_m = (img @ be.T) - (img @ ge.T)          # (N, k) per-pair margins
        cols[f"q_{q}_margin"] = pair_m.mean(axis=1).astype(np.float32)
        cols[f"q_{q}_disagreement"] = pair_m.std(axis=1).astype(np.float32)
    return cols


# ---- candidate gate signals: (gate_name -> raw_score array with higher=worse) --------
def candidate_gate_signals(table, qmargins: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assemble every candidate gate's raw signal (oriented so higher = more bad),
    from CLIP question margins + pixel features in `table`. calibrate_gates screens
    which survive. Returns {gate_name: raw_signal (N,)}."""
    import numpy as np
    sig: dict[str, np.ndarray] = {}
    # CLIP question gates: margin already higher=worse
    for q in CLIP_QUESTIONS:
        col = f"q_{q}_margin"
        if col in qmargins:
            sig[f"clip_{q}"] = np.asarray(qmargins[col], dtype=float)
    # pixel gates: flip low_bad features so higher=worse
    for defect, feats in PIXEL_GATES.items():
        for col, direction in feats:
            if col not in table.columns:
                continue
            v = table[col].to_numpy(dtype=float)
            sig[f"pixel_{defect}__{col}"] = (-v if direction == "low_bad" else v)
    return sig


# ---- rules-only defect gates: within-dataset percentile per defect (higher = worse) --
# Each gate reuses EXISTING signals; no thresholds are fit on any labels for RANKING.
#   pixel defects reuse simple_features percentile scores (already worse=high, in [0,1]).
#   CLIP defects use the question margin -> within-dataset percentile rank.
#   hybrid defects take the max of both (worst-of).
def _pct(series):
    import pandas as pd
    return series.rank(pct=True, method="average")


def defect_gate_table(table):
    """Return a DataFrame of per-defect gate scores in [0,1] (higher = worse), one
    column per defect gate. Pure rules: pixel percentiles + CLIP-margin percentiles.
    `table` must already contain simple_* scores and q_*_margin columns."""
    import pandas as pd
    g = pd.DataFrame(index=table.index)
    # pixel gates (already percentile-based worse=high from simple_features)
    if "simple_blur_score" in table:
        g["gate_blur"] = table["simple_blur_score"]
    if "simple_exposure_score" in table:
        g["gate_exposure"] = table["simple_exposure_score"]
    if "simple_resolution_score" in table:
        g["gate_resolution"] = table["simple_resolution_score"]
    # CLIP gates: margin (higher=worse) -> percentile
    for q, gate in [("crop", "gate_crop"), ("clutter", "gate_clutter"),
                    ("item_visibility", "gate_item_visibility"), ("tilt", "gate_tilt")]:
        col = f"q_{q}_margin"
        if col in table:
            g[gate] = _pct(table[col])
    # hybrid glare: worst of pixel glare percentile and CLIP glare margin percentile
    parts = []
    if "simple_glare_score" in table:
        parts.append(table["simple_glare_score"])
    if "q_glare_margin" in table:
        parts.append(_pct(table["q_glare_margin"]))
    if parts:
        g["gate_glare"] = pd.concat(parts, axis=1).max(axis=1)
    return g


ALL_GATES = ["gate_blur", "gate_exposure", "gate_resolution", "gate_crop",
             "gate_clutter", "gate_item_visibility", "gate_tilt", "gate_glare"]


def v4_score(table, kept_gates: list[str]):
    """v4 = max over KEPT defect-gate percentiles; reason = argmax gate. Rules-only,
    no label fitting in the ranking. Rows with all-NaN gates get NaN."""
    import numpy as np, pandas as pd
    g = defect_gate_table(table)
    use = [c for c in kept_gates if c in g.columns]
    if not use:
        raise ValueError("no kept gates present")
    sub = g[use]
    score = sub.max(axis=1)
    reason = pd.Series([None] * len(sub), index=sub.index, dtype=object)
    ok = sub.notna().any(axis=1)
    if ok.any():
        reason.loc[ok] = sub.loc[ok].idxmax(axis=1).str.replace("gate_", "", regex=False)
    return score, reason, g


def load_gates(path: Path) -> dict:
    """Load surviving gates + calibration (thresholds, robust scales) from json."""
    return json.loads(Path(path).read_text())


def hybrid_score(gate_signals: dict[str, np.ndarray], gates: dict):
    """Compute v4 hybrid score = max normalized threshold-exceedance over SURVIVING
    gates. `gates` = {gate_name: {"threshold": t, "scale": s}} (from calibrate_gates).
    Returns (score (N,), reason (N,) gate names)."""
    names = [g for g in gates if g in gate_signals]
    if not names:
        raise ValueError("no surviving gates present in signals")
    n = len(next(iter(gate_signals.values())))
    excess = np.full((n, len(names)), -np.inf)
    for j, g in enumerate(names):
        t = gates[g]["threshold"]; s = gates[g].get("scale", 1.0) or 1.0
        excess[:, j] = (gate_signals[g] - t) / s
    score = excess.max(axis=1)
    reason = np.array(names)[excess.argmax(axis=1)]
    return score.astype(np.float32), reason
