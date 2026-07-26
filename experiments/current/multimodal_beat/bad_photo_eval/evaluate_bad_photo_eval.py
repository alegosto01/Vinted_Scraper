"""Phase 9: merge blind human labels with the private candidate/score file and
compute precision@K metrics per method.

Inputs:
  --private-candidates  eval_candidates_private.csv (blind_id, item_id, search_names,
                         primary_search, title, image_path, is_repeat, selection_reasons,
                         v1_generic_clip_score, v2_typed_clip_score, clip_overall_margin,
                         simple_overall_score, clip_top_defect, simple_top_defect)
  --labels               blind_label_sheet.csv (blind_id, technical_quality,
                         hurts_listing_presentation, fixable_by_retake, defect_tags, notes)
  --out-dir               where the 7 output files are written

Higher method score = more likely bad. Main positive = technical_quality=="bad".
Strict positive = bad AND hurts_listing_presentation=="yes" AND fixable_by_retake=="yes".
Rows with technical_quality=="uncertain" are dropped before ranking/counting.
is_repeat==1 rows are excluded from all precision metrics (kept only for the
duplicate-review agreement check).

Usage:
  python evaluate_bad_photo_eval.py \
    --private-candidates data/evaluation/eval_candidates_private.csv \
    --labels data/evaluation/blind_label_sheet.csv \
    --out-dir data/evaluation
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {
    "v1": "v1_generic_clip_score",
    "v2": "v2_typed_clip_score",
    "v3": "clip_overall_margin",
    "simple": "simple_overall_score",
}
KS = [3, 5, 8]
DEFECT_TAGS = ["blur", "dark", "overexposed", "glare", "bad_crop", "extreme_tilt",
               "low_resolution", "noise", "clutter", "item_not_clear", "other"]
MIN_DEFECT_N = 5  # minimum labeled-positive rows for a defect tag before we trust its precision


# --------------------------------------------------------------------------- #
# loading / merge
# --------------------------------------------------------------------------- #

def load_and_merge(private_path: Path, labels_path: Path) -> pd.DataFrame:
    priv = pd.read_csv(private_path)
    labs = pd.read_csv(labels_path)
    merged = priv.merge(labs, on="blind_id", how="left", suffixes=("", "_label"))
    for col in ("technical_quality", "hurts_listing_presentation", "fixable_by_retake",
                "defect_tags", "notes"):
        if col not in merged.columns:
            merged[col] = np.nan
    merged["is_repeat"] = pd.to_numeric(merged.get("is_repeat", 0), errors="coerce").fillna(0).astype(int)
    merged["search_names"] = merged.get("search_names", "").fillna("").astype(str)
    merged["selection_reasons"] = merged.get("selection_reasons", "").fillna("").astype(str)
    merged["defect_tags"] = merged["defect_tags"].fillna("").astype(str)
    return merged


def is_labeled(df: pd.DataFrame) -> pd.Series:
    return df["technical_quality"].notna() & (df["technical_quality"].astype(str).str.strip() != "")


def is_uncertain(df: pd.DataFrame) -> pd.Series:
    return df["technical_quality"].astype(str).str.strip() == "uncertain"


def main_positive(df: pd.DataFrame) -> pd.Series:
    return df["technical_quality"].astype(str).str.strip() == "bad"


def strict_positive(df: pd.DataFrame) -> pd.Series:
    return (
        main_positive(df)
        & (df["hurts_listing_presentation"].astype(str).str.strip() == "yes")
        & (df["fixable_by_retake"].astype(str).str.strip() == "yes")
    )


def eval_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for precision metrics: labeled, non-uncertain, non-repeat."""
    keep = is_labeled(df) & ~is_uncertain(df) & (df["is_repeat"] == 0)
    return df[keep].copy()


# --------------------------------------------------------------------------- #
# precision@K
# --------------------------------------------------------------------------- #

def precision_at_k(pool: pd.DataFrame, score_col: str, k: int, pos_mask: pd.Series) -> float:
    """Rank pool by score_col desc, precision among top k. NaN if pool smaller than k."""
    if len(pool) < k or k <= 0:
        return float("nan")
    scores = pd.to_numeric(pool[score_col], errors="coerce")
    order = scores.sort_values(ascending=False, kind="mergesort").index
    top = order[:k]
    return float(pos_mask.loc[top].sum()) / k


def searches_present(pool: pd.DataFrame) -> list[str]:
    names = set()
    for s in pool["search_names"]:
        for n in str(s).split("|"):
            n = n.strip()
            if n:
                names.add(n)
    return sorted(names)


def rows_for_search(pool: pd.DataFrame, search: str) -> pd.DataFrame:
    mask = pool["search_names"].apply(lambda s: search in [x.strip() for x in str(s).split("|") if x.strip()])
    return pool[mask]


# --------------------------------------------------------------------------- #
# metric computation
# --------------------------------------------------------------------------- #

def compute_metrics(df: pd.DataFrame) -> dict:
    pool = eval_pool(df)
    pos = main_positive(pool)
    strict_pos = strict_positive(pool)

    n_unique_items = int((df["is_repeat"] == 0).sum())
    n_labeled = int(is_labeled(df).sum())
    n_uncertain = int(is_uncertain(df).sum())
    uncertain_rate = (n_uncertain / n_labeled) if n_labeled else float("nan")

    # random-bucket prevalence (labeled, non-uncertain, non-repeat, selection_reasons has ':random')
    random_mask = pool["selection_reasons"].str.contains(":random", na=False)
    random_pool = pool[random_mask]
    random_prevalence = float(pos.loc[random_pool.index].mean()) if len(random_pool) else float("nan")

    searches = searches_present(pool)

    by_method = {}
    by_search = {}
    by_defect = {}

    for m, col in METHODS.items():
        by_method[m] = {}
        # pooled (single global ranking)
        for k in KS:
            by_method[m][f"pooled_p@{k}"] = precision_at_k(pool, col, k, pos)

        # per-search
        per_search_p = {k: {} for k in KS}
        for s in searches:
            sp = rows_for_search(pool, s)
            spos = main_positive(sp)
            row_key = (s, m)
            by_search[row_key] = {"n": len(sp)}
            for k in KS:
                p = precision_at_k(sp, col, k, spos)
                per_search_p[k][s] = p
                by_search[row_key][f"p@{k}"] = p

        # macro average across searches with >= K labeled non-uncertain non-repeat items
        for k in KS:
            vals = [v for s, v in per_search_p[k].items() if not np.isnan(v)]
            by_method[m][f"macro_p@{k}"] = float(np.mean(vals)) if vals else float("nan")
            by_method[m][f"macro_p@{k}_n_searches"] = len(vals)

        # strict target (macro p@8 at least)
        strict_vals = []
        for s in searches:
            sp = rows_for_search(pool, s)
            sstrict = strict_positive(sp)
            v = precision_at_k(sp, col, 8, sstrict)
            if not np.isnan(v):
                strict_vals.append(v)
        by_method[m]["strict_macro_p@8"] = float(np.mean(strict_vals)) if strict_vals else float("nan")

        # lift over random-bucket prevalence
        macro8 = by_method[m]["macro_p@8"]
        if not np.isnan(macro8) and not np.isnan(random_prevalence) and random_prevalence > 0:
            by_method[m]["lift_over_random"] = float(macro8 / random_prevalence)
        else:
            by_method[m]["lift_over_random"] = float("nan")

        # per-search count of searches reaching p@8 >= 0.375 (used by decision rule)
        n_ge_0375 = sum(1 for v in per_search_p[8].values() if not np.isnan(v) and v >= 0.375)
        by_method[m]["n_searches_p@8_ge_0.375"] = n_ge_0375

        # by defect tag
        for tag in DEFECT_TAGS:
            tag_pos = pool["defect_tags"].str.contains(tag, na=False) & pos
            n_tag_pos = int(tag_pos.sum())
            key = (tag, m)
            if n_tag_pos < MIN_DEFECT_N or len(pool) < 8:
                by_defect[key] = {"n_positive": n_tag_pos, "precision@8": float("nan"),
                                   "status": "insufficient"}
                continue
            p = precision_at_k(pool, col, 8, tag_pos)
            by_defect[key] = {"n_positive": n_tag_pos, "precision@8": p, "status": "ok"}

    # duplicate-review agreement: item_ids appearing both as is_repeat==0 and is_repeat==1
    agreement = duplicate_agreement(df)

    return {
        "n_unique_items": n_unique_items,
        "n_labeled": n_labeled,
        "n_uncertain": n_uncertain,
        "uncertain_rate": uncertain_rate,
        "random_prevalence": random_prevalence,
        "n_random_bucket_labeled": int(len(random_pool)),
        "searches": searches,
        "by_method": by_method,
        "by_search": by_search,
        "by_defect": by_defect,
        "duplicate_agreement": agreement,
        "pool": pool,
    }


def duplicate_agreement(df: pd.DataFrame) -> dict:
    labeled = df[is_labeled(df) & ~is_uncertain(df)]
    orig = labeled[labeled["is_repeat"] == 0]
    rep = labeled[labeled["is_repeat"] == 1]
    common_items = set(orig["item_id"]) & set(rep["item_id"])
    if not common_items:
        return {"n_pairs": 0, "agreement_rate": float("nan")}
    o_map = orig.drop_duplicates("item_id").set_index("item_id")["technical_quality"]
    r_map = rep.drop_duplicates("item_id").set_index("item_id")["technical_quality"]
    agree = sum(1 for iid in common_items if o_map[iid] == r_map[iid])
    return {"n_pairs": len(common_items), "agreement_rate": agree / len(common_items)}


# --------------------------------------------------------------------------- #
# output writers
# --------------------------------------------------------------------------- #

def write_metrics_summary(metrics: dict, out_dir: Path) -> dict:
    by_method = metrics["by_method"]
    # "the better of {v1,v2}" — used only for the v3-vs-others comparison
    best_v1_v2_macro8 = max(
        (by_method[m]["macro_p@8"] for m in ("v1", "v2") if not np.isnan(by_method[m]["macro_p@8"])),
        default=float("nan"),
    )
    # "best CLIP method" (v1,v2,v3) — used for the prioritize-simple-baseline rule
    best_all_clip_macro8 = max(
        (by_method[m]["macro_p@8"] for m in ("v1", "v2", "v3") if not np.isnan(by_method[m]["macro_p@8"])),
        default=float("nan"),
    )

    v3 = by_method["v3"]
    v3_macro8 = v3["macro_p@8"]
    v3_lift = v3["lift_over_random"]
    v3_n_searches_ge = v3["n_searches_p@8_ge_0.375"]

    v3_worth_retaining = bool(
        not np.isnan(v3_macro8) and v3_macro8 >= 0.50
        and not np.isnan(v3_lift) and v3_lift >= 2.0
        and v3_n_searches_ge >= 4
    )
    v3_beats_best_v1v2 = None
    if not np.isnan(v3_macro8) and not np.isnan(best_v1_v2_macro8):
        v3_beats_best_v1v2 = bool(v3_macro8 - best_v1_v2_macro8 >= 0.10)

    simple_macro8 = by_method["simple"]["macro_p@8"]
    prioritize_simple = bool(
        not np.isnan(simple_macro8) and not np.isnan(best_all_clip_macro8)
        and simple_macro8 > best_all_clip_macro8
    )

    all_macro8 = [by_method[m]["macro_p@8"] for m in METHODS]
    any_reaches_50 = any((not np.isnan(v)) and v >= 0.50 for v in all_macro8)
    stop_prompt_tuning = bool(metrics["n_labeled"] > 0 and not any_reaches_50)

    summary = {
        "n_unique_items": metrics["n_unique_items"],
        "n_labeled": metrics["n_labeled"],
        "n_uncertain": metrics["n_uncertain"],
        "uncertain_rate": _clean(metrics["uncertain_rate"]),
        "random_prevalence": _clean(metrics["random_prevalence"]),
        "duplicate_agreement": {
            "n_pairs": metrics["duplicate_agreement"]["n_pairs"],
            "agreement_rate": _clean(metrics["duplicate_agreement"]["agreement_rate"]),
        },
        "macro_p@8_by_method": {m: _clean(by_method[m]["macro_p@8"]) for m in METHODS},
        "decision_rule": {
            "v3_worth_retaining": v3_worth_retaining if metrics["n_labeled"] else None,
            "v3_macro_p@8": _clean(v3_macro8),
            "v3_lift_over_random": _clean(v3_lift),
            "v3_n_searches_p@8_ge_0.375": v3_n_searches_ge,
            "v3_beats_best_of_v1_v2_by_0.10": v3_beats_best_v1v2,
            "best_of_v1_v2_macro_p@8": _clean(best_v1_v2_macro8),
            "prioritize_simple_baseline": prioritize_simple if metrics["n_labeled"] else None,
            "stop_prompt_tuning_collect_labels": stop_prompt_tuning,
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def write_metrics_by_method(metrics: dict, out_dir: Path):
    rows = []
    for m in METHODS:
        bm = metrics["by_method"][m]
        row = {"method": m}
        for k in KS:
            row[f"macro_p@{k}"] = bm[f"macro_p@{k}"]
            row[f"pooled_p@{k}"] = bm[f"pooled_p@{k}"]
        row["strict_macro_p@8"] = bm["strict_macro_p@8"]
        row["lift_over_random"] = bm["lift_over_random"]
        row["macro_p@8_n_searches"] = bm["macro_p@8_n_searches"]
        row["n_searches_p@8_ge_0.375"] = bm["n_searches_p@8_ge_0.375"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_method.csv", index=False)


def write_metrics_by_search(metrics: dict, out_dir: Path):
    rows = []
    for (s, m), d in metrics["by_search"].items():
        row = {"search": s, "method": m, "n": d["n"]}
        for k in KS:
            row[f"p@{k}"] = d[f"p@{k}"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["search", "method"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["search", "method", "n"] + [f"p@{k}" for k in KS])
    df.to_csv(out_dir / "metrics_by_search.csv", index=False)


def write_metrics_by_defect(metrics: dict, out_dir: Path):
    rows = []
    for (tag, m), d in metrics["by_defect"].items():
        rows.append({"defect_tag": tag, "method": m, "n_positive": d["n_positive"],
                     "precision@8": d["precision@8"], "status": d["status"]})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["defect_tag", "method"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["defect_tag", "method", "n_positive", "precision@8", "status"])
    df.to_csv(out_dir / "metrics_by_defect.csv", index=False)


def write_false_positives(df: pd.DataFrame, metrics: dict, out_dir: Path, top_n: int = 8):
    pool = metrics["pool"]
    rows = []
    if len(pool):
        pos = main_positive(pool)
        bad_labeled_good_or_notitem = pool["technical_quality"].astype(str).isin(["good", "not_item_photo"])
        for m, col in METHODS.items():
            scores = pd.to_numeric(pool[col], errors="coerce")
            order = scores.sort_values(ascending=False, kind="mergesort").index
            top = order[:top_n]
            fp = pool.loc[top]
            fp = fp[bad_labeled_good_or_notitem.loc[top]]
            for _, r in fp.iterrows():
                rows.append({
                    "method": m, "blind_id": r.get("blind_id"), "item_id": r.get("item_id"),
                    "score": r.get(col), "technical_quality": r.get("technical_quality"),
                    "title": r.get("title"), "image_path": r.get("image_path"),
                })
    out = pd.DataFrame(rows, columns=["method", "blind_id", "item_id", "score",
                                       "technical_quality", "title", "image_path"])
    out.to_csv(out_dir / "false_positives.csv", index=False)


def write_false_negatives_from_random(df: pd.DataFrame, metrics: dict, out_dir: Path, bottom_n: int = 8):
    pool = metrics["pool"]
    rows = []
    if len(pool):
        random_pool = pool[pool["selection_reasons"].str.contains(":random", na=False)]
        random_bad = random_pool[main_positive(random_pool)]
        for m, col in METHODS.items():
            if not len(random_bad):
                continue
            scores = pd.to_numeric(pool[col], errors="coerce")
            rank = scores.rank(ascending=False, method="min")
            for _, r in random_bad.iterrows():
                rk = rank.loc[r.name]
                if rk > len(pool) - bottom_n or rk > 0.75 * len(pool):
                    rows.append({
                        "method": m, "blind_id": r.get("blind_id"), "item_id": r.get("item_id"),
                        "score": r.get(col), "rank": int(rk), "pool_size": len(pool),
                        "title": r.get("title"), "image_path": r.get("image_path"),
                    })
    out = pd.DataFrame(rows, columns=["method", "blind_id", "item_id", "score", "rank",
                                       "pool_size", "title", "image_path"])
    out.to_csv(out_dir / "false_negatives_from_random.csv", index=False)


def write_results_md(metrics: dict, summary: dict, out_dir: Path):
    lines = ["# Bad-photo eval results\n"]
    if metrics["n_labeled"] == 0:
        lines.append("**No labels yet** - `blind_label_sheet.csv` has 0 labeled rows. "
                      "Fill in `technical_quality` (and the other label columns) and re-run "
                      "this script to get real metrics.\n")
        (out_dir / "results.md").write_text("\n".join(lines))
        return

    lines.append(f"- Unique items surfaced (is_repeat==0): **{metrics['n_unique_items']}**")
    lines.append(f"- Labeled rows: **{metrics['n_labeled']}**")
    lines.append(f"- Uncertain-label rate: **{_fmt(metrics['uncertain_rate'])}** "
                 f"({metrics['n_uncertain']} of {metrics['n_labeled']})")
    lines.append(f"- Random bad-photo prevalence (`:random` bucket): "
                 f"**{_fmt(metrics['random_prevalence'])}** "
                 f"(n={metrics['n_random_bucket_labeled']})")
    da = metrics["duplicate_agreement"]
    lines.append(f"- Duplicate-review label agreement: **{_fmt(da['agreement_rate'])}** "
                 f"(n_pairs={da['n_pairs']})\n")

    lines.append("## Primary metric: macro precision@8 across searches\n")
    lines.append("| method | macro p@8 | pooled p@8 | strict macro p@8 | lift over random | n searches |")
    lines.append("|---|---|---|---|---|---|")
    for m in METHODS:
        bm = metrics["by_method"][m]
        lines.append(f"| {m} | {_fmt(bm['macro_p@8'])} | {_fmt(bm['pooled_p@8'])} | "
                     f"{_fmt(bm['strict_macro_p@8'])} | {_fmt(bm['lift_over_random'])} | "
                     f"{bm['macro_p@8_n_searches']} |")

    lines.append("\n## precision@3 / @5 (macro)\n")
    lines.append("| method | macro p@3 | macro p@5 |")
    lines.append("|---|---|---|")
    for m in METHODS:
        bm = metrics["by_method"][m]
        lines.append(f"| {m} | {_fmt(bm['macro_p@3'])} | {_fmt(bm['macro_p@5'])} |")

    lines.append("\n## Decision rule\n")
    dr = summary["decision_rule"]
    lines.append(f"- v3 (clip_overall_margin) worth retaining: **{dr['v3_worth_retaining']}** "
                 f"(macro p@8={_fmt(dr['v3_macro_p@8'])}, lift={_fmt(dr['v3_lift_over_random'])}, "
                 f"n_searches p@8>=0.375: {dr['v3_n_searches_p@8_ge_0.375']})")
    lines.append(f"- v3 beats better of {{v1,v2}} by >= 0.10 absolute macro p@8: "
                 f"**{dr['v3_beats_best_of_v1_v2_by_0.10']}** "
                 f"(best of v1/v2 macro p@8={_fmt(dr['best_of_v1_v2_macro_p@8'])})")
    lines.append(f"- Prioritize simple baseline (simple > best CLIP method): "
                 f"**{dr['prioritize_simple_baseline']}**")
    if dr["stop_prompt_tuning_collect_labels"]:
        lines.append("- **STOP prompt tuning; collect labels for a small supervised model "
                     "(needs ~50 clear bad + 50 clear good).**")

    lines.append("\n## Caveats\n")
    lines.append("- Recall is not computed (candidate pool is a biased sample, not a random "
                 "draw of all listings).")
    lines.append("- Per-search and per-defect precision below the minimum sample size are "
                 "marked NaN/`insufficient` rather than reported as a misleadingly precise number.")
    lines.append("- Repeat rows (`is_repeat==1`) are excluded from all precision metrics; they "
                 "only feed the duplicate-review agreement check.")

    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.3f}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def run(private_candidates: Path, labels: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_merge(private_candidates, labels)
    metrics = compute_metrics(df)
    summary = write_metrics_summary(metrics, out_dir)
    write_metrics_by_method(metrics, out_dir)
    write_metrics_by_search(metrics, out_dir)
    write_metrics_by_defect(metrics, out_dir)
    write_false_positives(df, metrics, out_dir)
    write_false_negatives_from_random(df, metrics, out_dir)
    write_results_md(metrics, summary, out_dir)
    print(f"wrote outputs to {out_dir}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--private-candidates", default="data/evaluation/eval_candidates_private.csv")
    ap.add_argument("--labels", default="data/evaluation/blind_label_sheet.csv")
    ap.add_argument("--out-dir", default="data/evaluation")
    a = ap.parse_args()
    run(Path(a.private_candidates), Path(a.labels), Path(a.out_dir))
