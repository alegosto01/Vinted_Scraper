"""Final blind comparison of simple vs v3 vs v4 on the FRESH holdout.

This is the only place a validation claim may be made, and only after the holdout
label sheet is filled. Reuses the target/metric definitions from analyze_chatgpt_labels
so precision@K, macro, lift, and eligibility are computed identically.

Usage:
  python compare_final.py \
    --private data/holdout/holdout_candidates_private.csv \
    --labels  data/holdout/holdout_label_sheet.csv \
    --out-dir data/holdout/results
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_chatgpt_labels as A

# holdout carries three frozen ranking columns
FINAL_METHODS = {"simple": "simple_overall_score", "v3": "clip_overall_margin",
                 "v4": "v4_score"}
KS = [3, 5, 8, 10]


def _load(private: Path, labels: Path) -> pd.DataFrame:
    priv = pd.read_csv(private, low_memory=False)
    lab = pd.read_csv(labels, low_memory=False)
    assert lab["blind_id"].is_unique and priv["blind_id"].is_unique
    if set(lab.blind_id) != set(priv.blind_id):
        raise SystemExit("blind_id mismatch between holdout private and labels")
    m = priv.merge(lab, on="blind_id", validate="one_to_one")
    m["is_repeat"] = m["is_repeat"].fillna(0).astype(int)
    m["tq"] = m["technical_quality"].astype(str).str.strip().str.lower()
    m["search_list"] = m["search_names"].fillna("").astype(str).apply(
        lambda s: [x for x in s.split("|") if x])
    m["is_random"] = m["selection_reasons"].fillna("").astype(str).str.contains(":random")
    return m


def _eligible_T1(m):
    e = m[(m.is_repeat == 0) & (m.tq.isin(["bad", "good"]))].copy()
    e["pos"] = e.tq == "bad"
    return e


def _repeat_agreement(m):
    reps = m[m.is_repeat == 1]
    orig = m[m.is_repeat == 0].set_index("item_id")["tq"]
    if not len(reps):
        return None, 0
    ok = [reps.iloc[i].tq == orig.get(reps.iloc[i].item_id)
          for i in range(len(reps)) if reps.iloc[i].item_id in orig.index]
    return (round(float(np.mean(ok)), 3) if ok else None), len(ok)


def run(private: Path, labels: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    m = _load(private, labels)
    e = _eligible_T1(m)
    n_labeled = int((m.is_repeat == 0).sum())
    rnd = e[e.is_random]
    prevalence = float(rnd["pos"].mean()) if len(rnd) else None
    lift_ok = len(rnd) >= 10 and prevalence

    rows, per_search = [], {}
    for name, col in FINAL_METHODS.items():
        rec = {"method": name}
        ap, auc = A.method_scores(e, col)
        rec["ap"], rec["roc_auc"] = ap, auc
        for k in KS:
            p, tp, fp = A.precision_at_k(e, col, k)
            rec[f"pooled_p@{k}"] = p
            rec[f"lift@{k}"] = round(p / prevalence, 3) if (lift_ok and p) else None
        # per-search + macro
        ps = {}
        for s in A.SEARCHES:
            sub = e[e.search_list.apply(lambda l: s in l)]
            if len(sub):
                p8, _, _ = A.precision_at_k(sub, col, 8)
                p5, _, _ = A.precision_at_k(sub, col, 5)
                ps[s] = {"p@5": p5, "p@8": p8, "n": len(sub)}
        per_search[name] = ps
        for k in (5, 8):
            vals = [v[f"p@{k}"] for v in ps.values() if v["n"] >= k and v[f"p@{k}"] is not None]
            rec[f"macro_p@{k}"] = round(float(np.mean(vals)), 4) if vals else None
            rec[f"macro_p@{k}_n"] = len(vals)
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(out / "final_metrics_by_method.csv", index=False)

    agree, n_pairs = _repeat_agreement(m)
    summ = {
        "n_labeled_unique": n_labeled,
        "eligible_bad_good": int(len(e)), "bad": int(e["pos"].sum()),
        "random_eligible": int(len(rnd)), "random_prevalence": prevalence,
        "lift_available": bool(lift_ok),
        "repeat_agreement": agree, "repeat_pairs": n_pairs,
        "macro_p@8": {r["method"]: r["macro_p@8"] for r in rows},
        "pooled_p@8": {r["method"]: r["pooled_p@8"] for r in rows},
    }
    # verdict: does v4 beat simple and v3 on macro p@8?
    mp = summ["macro_p@8"]
    if all(mp.get(k) is not None for k in ("simple", "v3", "v4")):
        summ["v4_beats_simple"] = mp["v4"] > mp["simple"]
        summ["v4_beats_v3"] = mp["v4"] > mp["v3"]
        summ["best_method"] = max(mp, key=lambda k: mp[k])
    else:
        summ["verdict"] = "insufficient labels for macro p@8"
    json.dump(summ, open(out / "final_summary.json", "w"), indent=2, default=str)

    L = ["# Final holdout comparison — simple vs v3 vs v4\n",
         "Fresh blind holdout, not used for gate development. This is the validation set.\n",
         f"- labeled unique: {n_labeled} | eligible bad/good: {len(e)} (bad={int(e['pos'].sum())})",
         f"- random eligible: {len(rnd)} prevalence: {A._fmt(prevalence)} lift_available: {bool(lift_ok)}",
         f"- repeat agreement: {agree} ({n_pairs} pairs)\n",
         "| method | pooled P@5 | P@8 | macro P@8 (n) | AP | ROC AUC | lift@8 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['method']} | {A._fmt(r['pooled_p@5'])} | {A._fmt(r['pooled_p@8'])} "
                 f"| {A._fmt(r['macro_p@8'])} ({r['macro_p@8_n']}) | {A._fmt(r['ap'])} "
                 f"| {A._fmt(r['roc_auc'])} | {A._fmt(r['lift@8'])} |")
    L.append("\n## Per-search P@8\n| search | " + " | ".join(FINAL_METHODS) + " | n |")
    L.append("|---|" + "---|" * (len(FINAL_METHODS) + 1))
    for s in A.SEARCHES:
        cells, n = [], "—"
        for name in FINAL_METHODS:
            v = per_search[name].get(s)
            cells.append(A._fmt(v["p@8"]) if v else "—")
            if v:
                n = v["n"]
        L.append(f"| {s} | " + " | ".join(cells) + f" | {n} |")
    if "best_method" in summ:
        L.append(f"\n**Best by macro P@8: {summ['best_method']}** "
                 f"(v4 beats simple: {summ['v4_beats_simple']}, v4 beats v3: {summ['v4_beats_v3']}).")
    L.append("\n> Labels here should be independent/human where possible. If ChatGPT-"
             "labeled, this is a stronger-but-still-model check, not human ground truth.")
    (out / "final_results.md").write_text("\n".join(L))
    print("wrote", out, "| macro p@8:", summ["macro_p@8"])
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", default="data/holdout/holdout_candidates_private.csv")
    ap.add_argument("--labels", default="data/holdout/holdout_label_sheet.csv")
    ap.add_argument("--out-dir", default="data/holdout/results")
    a = ap.parse_args()
    run(Path(a.private), Path(a.labels), Path(a.out_dir))
