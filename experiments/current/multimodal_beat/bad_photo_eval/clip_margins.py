"""Phase 3: generic CLIP defect margins.

For each defect d with good/bad prompt sets:
    bad_sim  = mean(img @ bad_emb.T)      # over that defect's bad prompts
    good_sim = mean(img @ good_emb.T)     # over that defect's good prompts
    margin_d = bad_sim - good_sim
No softmax, no sigmoid, no calibration, no type selection. Text embeddings are encoded
once, L2-normalized and cached. Overall CLIP score = max over defect margins.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

from prompts import DEFECT_PROMPTS, DEFECTS

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "clip_text_cache"
EMBEDDING_MODEL = "openai/clip-vit-base-patch32"


def _prompt_hash() -> str:
    blob = json.dumps(DEFECT_PROMPTS, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def encode_prompts(cache: bool = True) -> dict[str, dict[str, np.ndarray]]:
    """Return {defect: {'good': (g,512), 'bad': (b,512)}}, L2-normalized. Cached by
    prompt-content hash so edits to prompts.py invalidate the cache automatically."""
    key = _prompt_hash()
    path = CACHE / f"defect_text_{key}.npz"
    if cache and path.exists():
        z = np.load(path)
        return {d: {"good": z[f"{d}__good"], "bad": z[f"{d}__bad"]} for d in DEFECTS}

    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(EMBEDDING_MODEL).eval()
    p = CLIPProcessor.from_pretrained(EMBEDDING_MODEL)

    def embed(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            inp = p(text=texts, return_tensors="pt", padding=True)
            t = m.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            v = m.text_projection(t.pooler_output).numpy()
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    out, flat = {}, {}
    for d in DEFECTS:
        g = embed(DEFECT_PROMPTS[d]["good"])
        b = embed(DEFECT_PROMPTS[d]["bad"])
        out[d] = {"good": g, "bad": b}
        flat[f"{d}__good"] = g
        flat[f"{d}__bad"] = b
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **flat)
    return out


def score_margins(img: np.ndarray, text: dict[str, dict[str, np.ndarray]] | None = None):
    """img: (N,512) L2-normalized image embeddings. Returns a dict of column arrays:
    clip_<d>_bad_similarity / _good_similarity / _margin for each defect, plus
    clip_overall_margin, clip_top_defect, clip_second_defect, clip_top_margin,
    clip_second_margin, clip_mean_positive_margin (diagnostic only)."""
    if text is None:
        text = encode_prompts()
    n = img.shape[0]
    margins = np.empty((n, len(DEFECTS)), dtype=np.float32)
    cols: dict[str, np.ndarray] = {}
    for j, d in enumerate(DEFECTS):
        bad_sim = (img @ text[d]["bad"].T).mean(axis=1)
        good_sim = (img @ text[d]["good"].T).mean(axis=1)
        margin = bad_sim - good_sim
        cols[f"clip_{d}_bad_similarity"] = bad_sim.astype(np.float32)
        cols[f"clip_{d}_good_similarity"] = good_sim.astype(np.float32)
        cols[f"clip_{d}_margin"] = margin.astype(np.float32)
        margins[:, j] = margin

    order = np.argsort(-margins, axis=1)          # descending margin per row
    defarr = np.array(DEFECTS)
    cols["clip_overall_margin"] = margins.max(axis=1)
    cols["clip_top_defect"] = defarr[order[:, 0]]
    cols["clip_second_defect"] = defarr[order[:, 1]]
    cols["clip_top_margin"] = np.take_along_axis(margins, order[:, :1], axis=1)[:, 0]
    cols["clip_second_margin"] = np.take_along_axis(margins, order[:, 1:2], axis=1)[:, 0]
    cols["clip_mean_positive_margin"] = np.where(margins > 0, margins, 0).sum(1) / \
        np.maximum((margins > 0).sum(1), 1)
    return cols
