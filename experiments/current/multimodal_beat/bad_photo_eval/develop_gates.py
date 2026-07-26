"""DEVELOPMENT gate screening for v4 (NOT final validation).

Uses the small HUMAN audit labels (46 images) as a development set to decide which
rules-only defect gates to KEEP in v4 and to set PROVISIONAL per-defect thresholds.
No thresholds are fit on ChatGPT pseudo-labels. Leave-one-search-out (LOSO) AUROC is
reported to flag overfitting on this tiny, enriched dev set.

The kept-gate list is written to surviving_gates.json and consumed by score v4. The
final simple-vs-v3-vs-v4 comparison MUST use a fresh blind holdout, not this dev set.

Usage:
  python develop_gates.py \
    --scored data/scored_with_questions.csv \
    --human  data/evaluation/chatgpt_results/human_audit_queue_reviewed.csv \
    --out-dir data/hybrid_dev
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hybrid

# lenient keep rule: dev set is tiny (~10 bad) and enriched -> screening, not proof.
KEEP_LOSO_AUROC = 0.55
KEEP_AUROC = 0.58
KEEP_MIN_POS = 4


def load_dev(scored: Path, human: Path):
    t = pd.read_csv(scored, low_memory=False)
    t["item_id"] = t["item_id"].astype(str)
    h = pd.read_csv(human)
    h["item_id"] = h["item_id"].astype(str)
    h["y"] = h["human_technical_quality"].astype(str).str.strip().str.lower()
    h = h[h["y"].isin(["bad", "good"])].copy()          # drop uncertain/not_item
    h["pos"] = (h["y"] == "bad").astype(int)
    gates = hybrid.defect_gate_table(t)
    dev = t[["item_id", "primary_search", "search_names"]].join(gates)
    dev = dev.merge(h[["item_id", "pos"]], on="item_id", how="inner")
    dev = dev.drop_duplicates("item_id")
    return dev, list(gates.columns)


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(s)
    y, s = np.asarray(y)[m], np.asarray(s)[m]
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, s))


def _loso_auroc(dev, gate):
    """Mean per-held-out-search AUROC: train searches unused (rules gate, no fit), just
    evaluate the gate's ranking within each held-out search separately, then average.
    Detects gates that only work in one search."""
    vals = []
    for s in dev["primary_search"].unique():
        sub = dev[dev["primary_search"] == s]
        a = _auroc(sub["pos"].to_numpy(), sub[gate].to_numpy())
        if a is not None:
            vals.append(a)
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def _provisional_threshold(dev, gate):
    """Youden-J point on the dev ROC (PROVISIONAL). Returns gate score cutoff."""
    from sklearn.metrics import roc_curve
    m = ~dev[gate].isna()
    y, s = dev.loc[m, "pos"].to_numpy(), dev.loc[m, gate].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return None
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def screen(scored: Path, human: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    dev, gate_cols = load_dev(scored, human)
    n_pos, n_neg = int(dev["pos"].sum()), int((dev["pos"] == 0).sum())
    rows = []
    for g in gate_cols:
        au = _auroc(dev["pos"].to_numpy(), dev[g].to_numpy())
        loso, n_s = _loso_auroc(dev, g)
        # top-k precision on dev (k=5) using this gate alone
        topk = dev.sort_values(g, ascending=False).head(5)
        p5 = float(topk["pos"].mean()) if len(topk) else None
        keep = bool(au is not None and loso is not None and au >= KEEP_AUROC and
                    loso >= KEEP_LOSO_AUROC and n_pos >= KEEP_MIN_POS)
        rows.append({"gate": g, "auroc": au, "loso_auroc": loso, "loso_searches": n_s,
                     "dev_p@5": p5, "provisional_threshold": _provisional_threshold(dev, g),
                     "keep": keep})
    rep = pd.DataFrame(rows).sort_values("loso_auroc", ascending=False, na_position="last")
    rep.to_csv(out / "gate_screening.csv", index=False)
    kept = rep[rep["keep"]]["gate"].tolist()
    if not kept:                       # fallback: keep top-3 by loso so v4 is defined
        kept = rep.dropna(subset=["loso_auroc"]).head(3)["gate"].tolist()
        fallback = True
    else:
        fallback = False
    surviving = {
        "kept_gates": kept,
        "fallback_used": fallback,
        "dev_n_pos": n_pos, "dev_n_neg": n_neg,
        "keep_rule": {"auroc": KEEP_AUROC, "loso_auroc": KEEP_LOSO_AUROC,
                      "min_pos": KEEP_MIN_POS},
        "per_gate": {r["gate"]: {k: r[k] for k in
                     ("auroc", "loso_auroc", "dev_p@5", "provisional_threshold", "keep")}
                     for r in rows},
        "note": "DEVELOPMENT screening on 46 enriched HUMAN labels. Provisional. "
                "Not final validation. Final comparison requires a fresh blind holdout.",
    }
    (out / "surviving_gates.json").write_text(json.dumps(surviving, indent=2, default=str))

    L = ["# v4 gate screening — DEVELOPMENT (not final)\n",
         f"Dev set: {n_pos} human-bad vs {n_neg} human-good (uncertain/not_item dropped). "
         "This set is tiny and enriched for hard cases; treat AUROC/precision as screening "
         "signals, **not** performance estimates.\n",
         "| gate | AUROC | LOSO AUROC | dev P@5 | keep |", "|---|---|---|---|---|"]
    for _, r in rep.iterrows():
        L.append(f"| {r.gate} | {r.auroc if r.auroc is None else round(r.auroc,3)} "
                 f"| {r.loso_auroc if r.loso_auroc is None else round(r.loso_auroc,3)} "
                 f"| {r['dev_p@5'] if r['dev_p@5'] is None else round(r['dev_p@5'],3)} "
                 f"| {r.keep} |")
    L.append(f"\n**Kept gates for v4:** {kept}" + (" (fallback: top-3 by LOSO)" if fallback else ""))
    L.append("\n> Provisional. Freeze v4 on these gates, then compare simple/v3/v4 on a "
             "NEW blind holdout. Do not cite these dev numbers as validation.")
    (out / "dev_report.md").write_text("\n".join(L))
    print("kept gates:", kept, "| fallback:", fallback)
    return surviving, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="data/scored_with_questions.csv")
    ap.add_argument("--human",
                    default="data/evaluation/chatgpt_results/human_audit_queue_reviewed.csv")
    ap.add_argument("--out-dir", default="data/hybrid_dev")
    a = ap.parse_args()
    screen(Path(a.scored), Path(a.human), Path(a.out_dir))
