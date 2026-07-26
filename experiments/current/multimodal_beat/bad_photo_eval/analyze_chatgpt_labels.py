"""Frozen evaluation of the bad-photo methods against ChatGPT pseudo-labels.

Reuses the FROZEN method scores already in eval_candidates_private.csv (v1/v2/v3/simple)
and merges the completed label sheet by blind_id. No prompt/threshold/model changes.

Three targets (evaluated separately, never collapsed):
  T1 technical_bad : bad(+) vs good(-)                  exclude uncertain, not_item_photo, repeats
  T2 invalid_image : not_item_photo(+) vs good|bad(-)   exclude uncertain, repeats
  T3 unusable      : bad|not_item_photo(+) vs good(-)   exclude uncertain, repeats

Labels are ChatGPT-generated PSEUDO-labels, not human ground truth.

Usage:
  python analyze_chatgpt_labels.py \
    --private data/evaluation/eval_candidates_private.csv \
    --labels  data/evaluation/blind_label_sheet_chatgpt.csv \
    --out-dir data/evaluation/chatgpt_results \
    --seed 24072026
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {"v1": "v1_generic_clip_score", "v2": "v2_typed_clip_score",
           "v3": "clip_overall_margin", "simple": "simple_overall_score"}
SEARCHES = ["prada", "gucci", "nike", "ps4", "griffati_donna_all", "griffati_uomo_all"]
KS = [3, 5, 8, 10]
PRIMARY_K = 8
DEFECT_TAGS = ["blur", "dark", "overexposed", "glare", "bad_crop", "extreme_tilt",
               "low_resolution", "noise", "clutter", "item_not_clear", "other"]
MIN_DEFECT_N = 5
LABEL_SOURCE = "chatgpt"


# ----------------------------- load & merge -----------------------------
def load_merged(private: Path, labels: Path) -> pd.DataFrame:
    priv = pd.read_csv(private, low_memory=False)
    lab = pd.read_csv(labels, low_memory=False)
    # never write selection metadata back into the (blind) label sheet; merge one-way.
    assert lab["blind_id"].is_unique, "duplicate blind_id in label sheet"
    assert priv["blind_id"].is_unique, "duplicate blind_id in private file"
    only_lab = set(lab["blind_id"]) - set(priv["blind_id"])
    only_priv = set(priv["blind_id"]) - set(lab["blind_id"])
    if only_lab or only_priv:
        raise SystemExit(f"blind_id mismatch: {len(only_lab)} label-only, "
                         f"{len(only_priv)} private-only. Merge must be exact.")
    m = priv.merge(lab, on="blind_id", how="inner", validate="one_to_one")
    assert len(m) == len(priv) == len(lab), "merge changed row count"
    m["is_repeat"] = m["is_repeat"].fillna(0).astype(int)
    m["tq"] = m["technical_quality"].astype(str).str.strip().str.lower()
    m["search_list"] = m["search_names"].fillna("").astype(str).apply(
        lambda s: [x for x in s.split("|") if x])
    m["tagset"] = m["defect_tags"].fillna("").astype(str).apply(
        lambda s: {t.strip() for t in s.split(";") if t.strip()})
    m["is_random"] = m["selection_reasons"].fillna("").astype(str).str.contains(":random")
    return m


# ----------------------------- validation -----------------------------
def validate(m: pd.DataFrame) -> dict:
    reps = m[m.is_repeat == 1]
    orig = m[m.is_repeat == 0]
    valid_tq = {"good", "bad", "uncertain", "not_item_photo"}
    invalid = m[~m.tq.isin(valid_tq)]
    rep_rows = []
    for _, r in reps.iterrows():
        match = orig[orig.item_id == r.item_id]
        if len(match):
            o = match.iloc[0]
            rep_rows.append({"repeat_blind_id": r.blind_id, "orig_blind_id": o.blind_id,
                             "item_id": r.item_id, "repeat_tq": r.tq, "orig_tq": o.tq,
                             "agree": bool(r.tq == o.tq), "image_path": r.image_path,
                             "title": r.get("title", "")})
    rep_df = pd.DataFrame(rep_rows)
    tag_counts = {}
    for t in DEFECT_TAGS:
        tag_counts[t] = int(orig.tagset.apply(lambda s: t in s).sum())
    rep = {
        "total_rows": int(len(m)),
        "unique_images": int(orig.item_id.nunique()),
        "hidden_repeat_rows": int(len(reps)),
        "technical_quality_counts": m.tq.value_counts().to_dict(),
        "defect_tag_counts_over_unique": tag_counts,
        "invalid_label_rows": int(len(invalid)),
        "duplicate_review_ids": int(m.blind_id.duplicated().sum()),
        "duplicate_item_ids_among_unique": int(orig.item_id.duplicated().sum()),
        "repeat_pairs": int(len(rep_df)),
        "repeat_agreement_rate": (round(float(rep_df.agree.mean()), 4)
                                  if len(rep_df) else None),
        "repeat_disagreements": int((~rep_df.agree).sum()) if len(rep_df) else 0,
    }
    return rep, rep_df


# ----------------------------- metric helpers -----------------------------
def eligible(m: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return non-repeat rows eligible for target, with a boolean 'pos' column."""
    base = m[m.is_repeat == 0].copy()
    if target == "T1":
        e = base[base.tq.isin(["bad", "good"])].copy()
        e["pos"] = e.tq == "bad"
    elif target == "T2":
        e = base[base.tq.isin(["not_item_photo", "good", "bad"])].copy()
        e["pos"] = e.tq == "not_item_photo"
    elif target == "T3":
        e = base[base.tq.isin(["bad", "not_item_photo", "good"])].copy()
        e["pos"] = e.tq.isin(["bad", "not_item_photo"])
    else:
        raise ValueError(target)
    return e


def precision_at_k(pool: pd.DataFrame, score_col: str, k: int):
    """Rank pool by score desc, precision among top-k. None if pool has < k rows."""
    if len(pool) < k:
        return None, None, None
    top = pool.sort_values(score_col, ascending=False).head(k)
    tp = int(top["pos"].sum())
    return tp / k, tp, k - tp


def method_scores(pool: pd.DataFrame, score_col: str):
    """AP + ROC AUC (secondary). None if only one class present."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = pool["pos"].astype(int).to_numpy()
    s = pool[score_col].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return None, None
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def per_search_pools(e: pd.DataFrame):
    """Yield (search, subpool) using AGGREGATED membership (item can be in several)."""
    for s in SEARCHES:
        sub = e[e.search_list.apply(lambda lst: s in lst)]
        if len(sub):
            yield s, sub


# ----------------------------- core evaluation -----------------------------
def evaluate_target(m: pd.DataFrame, target: str) -> dict:
    e = eligible(m, target)
    out = {"target": target, "eligible_rows": int(len(e)),
           "positives": int(e["pos"].sum()), "negatives": int((~e["pos"]).sum()),
           "per_method": {}, "per_search": {}, "macro": {}}
    # random prevalence (target-specific), from eligible non-repeat random rows
    rnd = e[e.is_random]
    out["random_eligible"] = int(len(rnd))
    out["random_prevalence"] = (round(float(rnd["pos"].mean()), 4) if len(rnd) else None)
    # too-small random denominator -> lift unavailable
    out["lift_available"] = bool(len(rnd) >= 10 and out["random_prevalence"])

    for name, col in METHODS.items():
        md = {"ap": None, "roc_auc": None}
        md["ap"], md["roc_auc"] = method_scores(e, col)
        for k in KS:
            p, tp, fp = precision_at_k(e, col, k)
            md[f"pooled_p@{k}"] = p
            md[f"pooled_tp@{k}"] = tp
            md[f"pooled_fp@{k}"] = fp
            if out["lift_available"] and p is not None:
                md[f"lift@{k}"] = round(p / out["random_prevalence"], 3)
            else:
                md[f"lift@{k}"] = None
        # per-search + macro
        persrch = {}
        for s, sub in per_search_pools(e):
            row = {}
            for k in KS:
                p, _, _ = precision_at_k(sub, col, k)
                row[f"p@{k}"] = p
            row["n_eligible"] = int(len(sub))
            persrch[s] = row
        out["per_search"][name] = persrch
        out["per_method"][name] = md

    # macro precision@K: mean over searches with >= K eligible rows
    for name in METHODS:
        macro = {}
        for k in KS:
            vals = [r[f"p@{k}"] for r in out["per_search"][name].values()
                    if r["n_eligible"] >= k and r[f"p@{k}"] is not None]
            macro[f"macro_p@{k}"] = round(float(np.mean(vals)), 4) if vals else None
            macro[f"macro_p@{k}_n_searches"] = len(vals)
        out["macro"][name] = macro
    return out, e


# ----------------------------- overlap + unique TPs -----------------------------
def topk_overlap_and_unique(e: pd.DataFrame, k: int = PRIMARY_K):
    tops = {name: set(e.sort_values(col, ascending=False).head(k)["blind_id"])
            for name, col in METHODS.items()}
    pos_ids = set(e[e["pos"]]["blind_id"])
    rows = []
    names = list(METHODS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = tops[a] & tops[b]
            union = tops[a] | tops[b]
            rows.append({"method_a": a, "method_b": b, "k": k,
                         "overlap_items": len(inter),
                         "jaccard": round(len(inter) / len(union), 3) if union else None})
    overlap = pd.DataFrame(rows)
    uniq = []
    for a in names:
        others = set().union(*[tops[b] for b in names if b != a])
        unique_tp = (tops[a] & pos_ids) - others
        uniq.append({"method": a, "k": k, "tp_in_topk": len(tops[a] & pos_ids),
                     "unique_tp_not_in_other_topk": len(unique_tp),
                     "unique_tp_ids": ";".join(sorted(unique_tp))})
    return overlap, pd.DataFrame(uniq)


# ----------------------------- defect diagnostics -----------------------------
def defect_diagnostics(m: pd.DataFrame) -> pd.DataFrame:
    e = eligible(m, "T1")           # bad/good pool
    rows = []
    for tag in DEFECT_TAGS:
        pos_mask = e.tagset.apply(lambda s: tag in s) & e["pos"]
        n_pos = int(pos_mask.sum())
        low_sample = n_pos < MIN_DEFECT_N
        for name, col in METHODS.items():
            tagged = e[pos_mask][col]
            untagged = e[~pos_mask][col]
            # precision@K restricted so positives = this tag
            sub = e.copy()
            sub["pos"] = pos_mask.values
            p8, tp8, _ = precision_at_k(sub, col, PRIMARY_K)
            rows.append({
                "defect": tag, "method": name, "n_positive": n_pos,
                "low_sample": low_sample,
                "median_score_tagged": round(float(tagged.median()), 4) if len(tagged) else None,
                "median_score_untagged": round(float(untagged.median()), 4) if len(untagged) else None,
                f"precision@{PRIMARY_K}": p8, f"tp@{PRIMARY_K}": tp8})
    return pd.DataFrame(rows)


# ----------------------------- error-analysis files -----------------------------
ERR_COLS = ["blind_id", "item_id", "image_path", "search_names", "title", "tq",
            "defect_tags", "v1_generic_clip_score", "v2_typed_clip_score",
            "clip_overall_margin", "simple_overall_score", "selection_reasons",
            "is_repeat", "clip_top_defect", "simple_top_defect"]


def _with_ranks(e: pd.DataFrame) -> pd.DataFrame:
    d = e.copy()
    for name, col in METHODS.items():
        d[f"rank_{name}"] = d[col].rank(ascending=False, method="min").astype(int)
    return d


def error_files(m: pd.DataFrame, out: Path, k_region: int = 10):
    e = _with_ranks(eligible(m, "T1"))
    cols = [c for c in ERR_COLS if c in e.columns] + [f"rank_{n}" for n in METHODS]
    # v3 false positives: v3 ranks high (top region) but labeled good
    fp = e[(e["rank_v3"] <= k_region) & (~e["pos"])].sort_values("rank_v3")
    fp[cols].to_csv(out / "false_positives_v3.csv", index=False)
    # v3 false negatives: labeled bad but ranked low by v3 (relative to this pool)
    fn = e[e["pos"]].sort_values("rank_v3", ascending=False)
    fn[cols].to_csv(out / "false_negatives_v3.csv", index=False)
    # disagreements: large rank gap between v3 and other method
    for other in ["simple", "v1", "v2"]:
        d = e.copy()
        d["rank_gap"] = (d[f"rank_{other}"] - d["rank_v3"]).abs()
        d = d.sort_values("rank_gap", ascending=False)
        d[cols + ["rank_gap"]].head(40).to_csv(
            out / f"v3_vs_{other}_disagreements.csv", index=False)
    return fp, fn


# ----------------------------- human-audit queue -----------------------------
AUDIT_EDIT = ["human_technical_quality", "human_defect_tags", "human_notes", "human_reviewed"]


def build_audit_queue(m: pd.DataFrame, out: Path, seed: int, target_n=(50, 80)):
    rng = np.random.default_rng(seed)
    e = _with_ranks(eligible(m, "T1"))
    picks = {}          # blind_id -> reason

    def add(df, reason):
        for bid in df["blind_id"]:
            picks.setdefault(bid, reason)

    # 1 v3 false positives in top region
    add(e[(e.rank_v3 <= 10) & (~e.pos)].sort_values("rank_v3"), "v3_false_positive")
    # 2 uncertain rows (from full non-repeat set, not just T1 pool)
    base = m[m.is_repeat == 0]
    add(base[base.tq == "uncertain"], "uncertain")
    # 3 repeat disagreements
    reps = m[m.is_repeat == 1]
    orig = m[m.is_repeat == 0].set_index("item_id")["tq"]
    dis = reps[reps.apply(lambda r: orig.get(r.item_id) not in (None, r.tq), axis=1)]
    add(dis, "repeat_disagreement")
    # 4 top v3-vs-{simple,v1,v2} disagreements
    for other in ["simple", "v1", "v2"]:
        d = e.copy(); d["gap"] = (d[f"rank_{other}"] - d.rank_v3).abs()
        add(d.sort_values("gap", ascending=False).head(6), f"disagree_v3_{other}")
    # 5 v3 top per search
    for s, sub in per_search_pools(e):
        add(sub.sort_values("clip_overall_margin", ascending=False).head(3),
            f"v3_top_{s}")
    # 6 balanced random sample
    rnd = base[base.is_random]
    if len(rnd):
        add(rnd.sample(min(8, len(rnd)), random_state=seed), "random_control")
    # 7 one example per defect tag
    for tag in DEFECT_TAGS:
        ex = base[base.tagset.apply(lambda s: tag in s)]
        if len(ex):
            add(ex.head(1), f"defect_{tag}")

    rank_cols = {f"rank_{n}": dict(zip(e.blind_id, e[f"rank_{n}"])) for n in METHODS}
    q = m[m.blind_id.isin(picks)].copy()
    for rc, mp in rank_cols.items():
        q[rc] = q.blind_id.map(mp).fillna(-1).astype(int)
    q["audit_reason"] = q.blind_id.map(picks)
    q = q.drop_duplicates("item_id").sort_values("audit_reason").reset_index(drop=True)
    # trim toward target while keeping conclusion-critical reasons first
    prio = ["v3_false_positive", "repeat_disagreement", "disagree_v3_simple",
            "disagree_v3_v1", "disagree_v3_v2", "uncertain"]
    q["prio"] = q.audit_reason.apply(lambda r: 0 if any(r.startswith(p) for p in prio) else 1)
    q = q.sort_values(["prio", "audit_reason", "blind_id"])
    if len(q) > target_n[1]:
        q = q.head(target_n[1])
    cols = [c for c in ERR_COLS if c in q.columns] + \
           [f"rank_{n}" for n in METHODS] + ["audit_reason"]
    aq = q[cols].copy()
    aq["original_label_source"] = LABEL_SOURCE
    for c in AUDIT_EDIT:
        aq[c] = ""
    aq.to_csv(out / "human_audit_queue.csv", index=False)
    _audit_gallery(aq, out / "human_audit_gallery.html")
    return aq


def _audit_gallery(aq: pd.DataFrame, path: Path):
    tags = DEFECT_TAGS
    tq_opts = ["good", "bad", "uncertain", "not_item_photo"]
    cards = []
    for _, r in aq.iterrows():
        scores = (f"v1={r.v1_generic_clip_score:.3f} v2={r.v2_typed_clip_score:.3f} "
                  f"v3={r.clip_overall_margin:.3f} simple={r.simple_overall_score:.3f}")
        ranks = " ".join(f"{n}#{int(r[f'rank_{n}'])}" for n in METHODS)
        bid = r.blind_id
        tq_radios = "".join(
            f'<label><input type=radio name="tq_{bid}" value="{o}"> {o}</label>'
            for o in tq_opts)
        tag_boxes = "".join(
            f'<label><input type=checkbox name="tag_{bid}" value="{t}"> {t}</label>'
            for t in tags)
        cards.append(f"""<div class=card data-bid="{bid}" data-item="{r.item_id}">
  <img src="file://{r.image_path}" loading=lazy>
  <div class=meta>
    <b>{bid}</b> <span class=reason>{r.audit_reason}</span><br>
    <span class=tq>chatgpt: {r.tq}</span> · tags: {r.defect_tags or '-'}<br>
    <small>{r.search_names} · {str(r.title)[:60]}</small><br>
    <small>{scores}</small><br><small>ranks: {ranks}</small>
  </div>
  <div class=form>
    <div class=row><span class=lbl>your call:</span> {tq_radios}</div>
    <div class=row><span class=lbl>defects:</span> {tag_boxes}</div>
    <div class=row><input class=notes name="notes_{bid}" placeholder="notes"></div>
  </div></div>""")
    html = f"""<!doctype html><meta charset=utf-8>
<title>v3 human-audit queue (ChatGPT pseudo-labels)</title>
<style>body{{font:14px sans-serif;background:#111;color:#eee;margin:16px}}
.bar{{position:sticky;top:0;background:#000;padding:10px;margin:-16px -16px 10px;z-index:9}}
button{{font:14px sans-serif;padding:8px 14px;background:#2b7;border:0;border-radius:6px;
color:#000;cursor:pointer}} #prog{{margin-left:12px;color:#9cf}}
.card{{display:inline-block;width:320px;vertical-align:top;margin:8px;background:#1c1c1c;
border-radius:8px;padding:8px}} .card img{{width:100%;border-radius:4px}}
.reason{{color:#f90}} .tq{{color:#9cf}} .meta{{margin-top:6px;line-height:1.4}}
.form{{margin-top:6px;font-size:12px}} .row{{margin:3px 0}} .lbl{{color:#888;margin-right:4px}}
.form label{{margin-right:8px;white-space:nowrap;display:inline-block}}
.notes{{width:95%;background:#111;border:1px solid #444;color:#eee;padding:3px}}
.card.done{{outline:2px solid #2b7}}</style>
<div class=bar>
  <button onclick="dl()">Download CSV</button>
  <span id=prog>0 / {len(aq)} reviewed</span>
  <span style="color:#888;margin-left:12px">labels shown are ChatGPT pseudo-labels — record YOUR call</span>
</div>
{''.join(cards)}
<script>
function state(){{let done=0;document.querySelectorAll('.card').forEach(c=>{{
  let bid=c.dataset.bid;
  let tq=c.querySelector('input[name="tq_'+bid+'"]:checked');
  if(tq){{c.classList.add('done');done++;}}else{{c.classList.remove('done');}}
}});document.getElementById('prog').textContent=done+' / '+document.querySelectorAll('.card').length+' reviewed';}}
document.addEventListener('change',state);document.addEventListener('input',state);
function esc(s){{s=(s==null?'':''+s);return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}}
function dl(){{
  let rows=[['blind_id','item_id','human_technical_quality','human_defect_tags','human_notes','human_reviewed']];
  document.querySelectorAll('.card').forEach(c=>{{
    let bid=c.dataset.bid;
    let tq=c.querySelector('input[name="tq_'+bid+'"]:checked');
    let tags=[...c.querySelectorAll('input[name="tag_'+bid+'"]:checked')].map(x=>x.value).join(';');
    let notes=c.querySelector('input[name="notes_'+bid+'"]').value;
    let reviewed=tq?'yes':'';
    rows.push([bid,c.dataset.item,tq?tq.value:'',tags,notes,reviewed].map(esc));
  }});
  let csv=rows.map(r=>r.join(',')).join('\\n');
  let a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));
  a.download='human_audit_queue_reviewed.csv';a.click();
}}
state();
</script>"""
    path.write_text(html)


# ----------------------------- results.md -----------------------------
def _fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _df_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_results(m, val, results, defects, overlap, uniq, out: Path):
    L = []
    L.append("# Bad-photo evaluation — ChatGPT pseudo-labels\n")
    L.append("> Labels are **ChatGPT-generated pseudo-labels**, not independent human "
             "ground truth. The sample was enriched via method rankings, so **dataset "
             "recall is not computable**. Do not calibrate thresholds or train on these "
             "labels without a fresh holdout. Some images were judged at contact-sheet "
             "resolution, so blur/noise/subtle-glare labels are less reliable.\n")

    L.append("## Dataset summary\n")
    for k, v in val.items():
        L.append(f"- {k}: {v}")
    L.append("")

    for tkey, tname in [("T1", "Technical bad photo (bad vs good)"),
                        ("T2", "Invalid first image (not_item_photo vs good/bad)"),
                        ("T3", "Operationally unusable (bad|not_item_photo vs good)")]:
        r = results[tkey]
        L.append(f"\n## Target {tkey}: {tname}\n")
        L.append(f"eligible={r['eligible_rows']} positives={r['positives']} "
                 f"negatives={r['negatives']} | random_eligible={r['random_eligible']} "
                 f"random_prevalence={_fmt(r['random_prevalence'])} "
                 f"lift_available={r['lift_available']}\n")
        L.append("| method | pooled P@3 | P@5 | P@8 | P@10 | AP | ROC AUC | macro P@8 (n) | lift@8 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name in METHODS:
            md = r["per_method"][name]; mac = r["macro"][name]
            L.append(f"| {name} | {_fmt(md['pooled_p@3'])} | {_fmt(md['pooled_p@5'])} "
                     f"| {_fmt(md['pooled_p@8'])} | {_fmt(md['pooled_p@10'])} "
                     f"| {_fmt(md['ap'])} | {_fmt(md['roc_auc'])} "
                     f"| {_fmt(mac['macro_p@8'])} ({mac['macro_p@8_n_searches']}) "
                     f"| {_fmt(md['lift@8'])} |")
        if tkey == "T1":
            L.append("\n### T1 per-search P@8\n")
            L.append("| search | " + " | ".join(METHODS) + " | n_eligible |")
            L.append("|---|" + "---|" * (len(METHODS) + 1))
            for s in SEARCHES:
                cells = []
                nelig = "—"
                for name in METHODS:
                    ps = r["per_search"][name].get(s)
                    if ps:
                        cells.append(_fmt(ps["p@8"])); nelig = ps["n_eligible"]
                    else:
                        cells.append("—")
                L.append(f"| {s} | " + " | ".join(cells) + f" | {nelig} |")

    L.append("\n## Defect-level diagnostics (T1 pool)\n")
    L.append("| defect | n_pos | low_sample | method | med_tagged | med_untagged | P@8 |")
    L.append("|---|---|---|---|---|---|---|")
    for _, d in defects.sort_values(["n_positive", "defect"], ascending=[False, True]).iterrows():
        L.append(f"| {d.defect} | {d.n_positive} | {d.low_sample} | {d.method} "
                 f"| {_fmt(d.median_score_tagged)} | {_fmt(d.median_score_untagged)} "
                 f"| {_fmt(d[f'precision@{PRIMARY_K}'])} |")

    L.append("\n## Top-8 method overlap & unique true positives (T1)\n")
    L.append(_df_md(overlap))
    L.append("")
    L.append(_df_md(uniq[["method","tp_in_topk","unique_tp_not_in_other_topk"]]))

    # interpretation
    r1 = results["T1"]
    macros = {n: r1["macro"][n]["macro_p@8"] for n in METHODS}
    best_legacy = max([("v1", macros["v1"] or 0), ("v2", macros["v2"] or 0),
                       ("simple", macros["simple"] or 0)], key=lambda x: x[1])
    v3m = macros["v3"] or 0
    dr = {
        "v3_macro_p@8": macros["v3"],
        "v3_lift@8": r1["per_method"]["v3"]["lift@8"],
        "v3_p@8_ge_0.375_n_searches": sum(
            1 for s in SEARCHES
            if (r1["per_search"]["v3"].get(s, {}).get("p@8") or 0) >= 0.375),
        "criterion_macro_p8_ge_0.50": (v3m >= 0.50),
        "criterion_lift_ge_2.0": (r1["per_method"]["v3"]["lift@8"] or 0) >= 2.0
        if r1["lift_available"] else "unresolved",
        "criterion_p8_ge_0.375_in_>=4_searches": None,
        "beats_best_legacy_by_0.10": (v3m - best_legacy[1]) >= 0.10,
        "best_legacy": best_legacy[0], "best_legacy_macro_p@8": best_legacy[1],
    }
    dr["criterion_p8_ge_0.375_in_>=4_searches"] = dr["v3_p@8_ge_0.375_n_searches"] >= 4
    dr["v3_validated"] = bool(dr["criterion_macro_p8_ge_0.50"] and
                              dr["criterion_lift_ge_2.0"] is True and
                              dr["criterion_p8_ge_0.375_in_>=4_searches"])
    L.append("\n## Decision rule (v3)\n")
    for k, v in dr.items():
        L.append(f"- {k}: {v}")
    L.append("\n### Limitations\n"
             "- ChatGPT pseudo-labels, not human ground truth; may share CLIP biases.\n"
             "- Enriched sample -> no full-dataset recall.\n"
             "- No threshold/model fitting on these labels; needs fresh holdout.\n"
             "- Contact-sheet resolution weakens blur/noise/subtle-glare labels.\n")
    out.joinpath("results.md").write_text("\n".join(L))
    return dr


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", default="data/evaluation/eval_candidates_private.csv")
    ap.add_argument("--labels", default="data/evaluation/blind_label_sheet_chatgpt.csv")
    ap.add_argument("--out-dir", default="data/evaluation/chatgpt_results")
    ap.add_argument("--seed", type=int, default=24072026)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    m = load_merged(Path(a.private), Path(a.labels))
    m.to_csv(out / "evaluation_merged_private.csv", index=False)
    val, rep_df = validate(m)
    (out / "repeat_consistency.csv").write_text(rep_df.to_csv(index=False))
    if len(rep_df) and (~rep_df.agree).any():
        rep_df[~rep_df.agree].to_csv(out / "repeat_disagreements.csv", index=False)

    results = {}
    for t in ("T1", "T2", "T3"):
        results[t], _ = evaluate_target(m, t)

    eT1 = eligible(m, "T1")
    overlap, uniq = topk_overlap_and_unique(eT1, PRIMARY_K)
    overlap.to_csv(out / "method_topk_overlap.csv", index=False)
    uniq.to_csv(out / "method_unique_tps.csv", index=False)
    defects = defect_diagnostics(m)
    defects.to_csv(out / "metrics_by_defect.csv", index=False)

    # per-method / per-search metric CSVs for T1
    rows = []
    for name in METHODS:
        md = results["T1"]["per_method"][name]; mac = results["T1"]["macro"][name]
        rows.append({"method": name, **{k: md[k] for k in md}, **mac})
    pd.DataFrame(rows).to_csv(out / "metrics_by_method_T1.csv", index=False)
    srows = []
    for name in METHODS:
        for s, r in results["T1"]["per_search"][name].items():
            srows.append({"method": name, "search": s, **r})
    pd.DataFrame(srows).to_csv(out / "metrics_by_search_T1.csv", index=False)

    error_files(m, out)
    build_audit_queue(m, out, a.seed)
    dr = write_results(m, val, results, defects, overlap, uniq, out)

    json.dump({"validation": val, "decision_rule": dr,
               "targets": {t: {"macro": results[t]["macro"],
                               "random_prevalence": results[t]["random_prevalence"],
                               "lift_available": results[t]["lift_available"]}
                           for t in results}},
              open(out / "metrics_summary.json", "w"), indent=2, default=str)
    print("wrote analysis to", out)
    print("v3 macro P@8:", dr["v3_macro_p@8"], "| validated:", dr["v3_validated"])


if __name__ == "__main__":
    main()
