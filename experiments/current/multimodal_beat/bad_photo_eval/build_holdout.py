"""Build a FRESH BLIND HOLDOUT for the final simple-vs-v3-vs-v4 comparison.

Excludes every item_id already used in the 244-row development set, selects new items
by the three frozen ranking methods + a random control, and emits a blind label sheet,
a fillable blind gallery, and a ChatGPT-labeling bundle. Deterministic per --seed.

Usage:
  python build_holdout.py --scores data/scored_with_v4.csv \
    --exclude data/evaluation/eval_candidates_private.csv \
    --per-method-search 3 --random-per-search 2 --repeat-fraction 0.10 \
    --seed 20260725 --out-dir data/holdout
"""
from __future__ import annotations
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEARCHES = ["prada", "gucci", "nike", "ps4", "griffati_donna_all", "griffati_uomo_all"]
METHOD_COL = {"simple": "simple_overall_score", "v3": "clip_overall_margin", "v4": "v4_score"}
DEFECT_TAGS = ["blur", "dark", "overexposed", "glare", "bad_crop", "extreme_tilt",
               "low_resolution", "noise", "clutter", "item_not_clear", "other"]
TQ_OPTS = ["good", "bad", "uncertain", "not_item_photo"]
YN = ["yes", "no", "uncertain"]


def in_search(row, s):
    return s in str(row["search_names"]).split("|")


def build(a):
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.scores, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)
    excl = set(pd.read_csv(a.exclude, usecols=["item_id"])["item_id"].astype(str))
    pool = df[(df["image_available"] == 1) & df["image_path"].notna() &
              (~df["item_id"].isin(excl))].copy()
    if a.max_items:
        pool = pool.head(a.max_items)

    reasons: dict[str, list[str]] = {}
    rep_rows = []

    def mark(item, reason):
        reasons.setdefault(item, [])
        if reason not in reasons[item]:
            reasons[item].append(reason)

    for s in SEARCHES:
        sub = pool[pool.apply(lambda r: in_search(r, s), axis=1)]
        if sub.empty:
            continue
        top_union = set()
        for name, col in METHOD_COL.items():
            k = max(a.per_method_search * 3, 50)
            top_union |= set(sub.nlargest(k, col)["item_id"])
            for iid in sub.nlargest(a.per_method_search, col)["item_id"]:
                mark(iid, f"{s}:{name}")
        outside = sub[~sub["item_id"].isin(top_union)]
        if len(outside):
            take = min(a.random_per_search, len(outside))
            pick = outside.sample(take, random_state=int(rng.integers(1 << 31)))
            for iid in pick["item_id"]:
                mark(iid, f"{s}:random")

    sel = df[df["item_id"].isin(reasons)].drop_duplicates("item_id").copy()
    sel["selection_reasons"] = sel["item_id"].map(lambda i: ";".join(sorted(reasons[i])))
    sel["is_repeat"] = 0
    n_unique = len(sel)

    # repeats: reproducible sample
    n_rep = math.ceil(a.repeat_fraction * n_unique)
    rep_ids = sel.sample(n_rep, random_state=int(rng.integers(1 << 31)))["item_id"].tolist() \
        if n_rep else []
    reps = sel[sel["item_id"].isin(rep_ids)].copy()
    reps["is_repeat"] = 1

    full = pd.concat([sel, reps], ignore_index=True)
    # deterministic blind_id assignment in shuffled order
    order = rng.permutation(len(full))
    full = full.iloc[order].reset_index(drop=True)
    full["blind_id"] = [f"H{i+1:04d}" for i in range(len(full))]

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    keep = ["blind_id", "item_id", "search_names", "primary_search", "title",
            "image_path", "is_repeat", "selection_reasons",
            "simple_overall_score", "clip_overall_margin", "v4_score",
            "v4_top_defect", "clip_top_defect", "simple_top_defect"]
    private = full[[c for c in keep if c in full.columns]]
    private.to_csv(out / "holdout_candidates_private.csv", index=False)

    blind = full[["blind_id", "image_path"]].copy()
    for c in ["technical_quality", "hurts_listing_presentation", "fixable_by_retake",
              "defect_tags", "notes"]:
        blind[c] = ""
    blind.to_csv(out / "holdout_label_sheet.csv", index=False)

    json.dump({"seed": a.seed, "searches": SEARCHES,
               "per_method_search": a.per_method_search,
               "random_per_search": a.random_per_search,
               "repeat_fraction": a.repeat_fraction,
               "n_unique": int(n_unique), "n_repeats": int(len(reps)),
               "n_total": int(len(full)), "excluded_count": len(excl),
               "created_at": datetime.now(timezone.utc).isoformat()},
              open(out / "holdout_manifest.json", "w"), indent=2)

    _blind_gallery(full, out / "holdout_blind_gallery.html")
    _chatgpt_bundle(full, out / "chatgpt_bundle")
    print(f"unique={n_unique} repeats={len(reps)} total={len(full)} "
          f"excluded={len(excl)} -> {out}")
    return full


def _blind_gallery(full, path: Path):
    cards = []
    for _, r in full.iterrows():
        bid = r.blind_id
        tq = "".join(f'<label><input type=radio name="tq_{bid}" value="{o}"> {o}</label>'
                     for o in TQ_OPTS)
        hurt = "".join(f'<label><input type=radio name="hurt_{bid}" value="{o}"> {o}</label>'
                       for o in YN)
        fix = "".join(f'<label><input type=radio name="fix_{bid}" value="{o}"> {o}</label>'
                      for o in YN)
        tags = "".join(f'<label><input type=checkbox name="tag_{bid}" value="{t}"> {t}</label>'
                       for t in DEFECT_TAGS)
        cards.append(f"""<div class=card data-bid="{bid}">
  <img src="file://{r.image_path}" loading=lazy>
  <div class=f><b>{bid}</b>
    <div class=row><span class=l>quality</span>{tq}</div>
    <div class=row><span class=l>hurts?</span>{hurt}</div>
    <div class=row><span class=l>fixable?</span>{fix}</div>
    <div class=row><span class=l>defects</span>{tags}</div>
    <div class=row><input class=notes name="notes_{bid}" placeholder="notes"></div>
  </div></div>""")
    html = f"""<!doctype html><meta charset=utf-8><title>Holdout blind labeling</title>
<style>body{{font:14px sans-serif;background:#111;color:#eee;margin:16px}}
.bar{{position:sticky;top:0;background:#000;padding:10px;margin:-16px -16px 10px;z-index:9}}
button{{padding:8px 14px;background:#2b7;border:0;border-radius:6px;color:#000;cursor:pointer}}
#p{{margin-left:12px;color:#9cf}}
.card{{display:inline-block;width:320px;vertical-align:top;margin:8px;background:#1c1c1c;
border-radius:8px;padding:8px}} .card img{{width:100%;border-radius:4px}}
.f{{font-size:12px;margin-top:6px}} .row{{margin:3px 0}} .l{{color:#888;margin-right:4px}}
.f label{{margin-right:6px;white-space:nowrap;display:inline-block}}
.notes{{width:95%;background:#111;border:1px solid #444;color:#eee}}
.card.done{{outline:2px solid #2b7}}</style>
<div class=bar><button onclick="dl()">Download CSV</button><span id=p></span>
<span style="color:#888;margin-left:12px">Judge ONLY first-image technical quality — not product/price/brand.</span></div>
{''.join(cards)}
<script>
function st(){{let d=0,c=document.querySelectorAll('.card');c.forEach(x=>{{
let b=x.dataset.bid,t=x.querySelector('input[name="tq_'+b+'"]:checked');
if(t){{x.classList.add('done');d++}}else x.classList.remove('done')}});
document.getElementById('p').textContent=d+' / '+c.length+' done'}}
document.addEventListener('change',st);document.addEventListener('input',st);
function e(s){{s=(s==null?'':''+s);return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}}
function g(b,n){{let x=document.querySelector('input[name="'+n+'_'+b+'"]:checked');return x?x.value:''}}
function dl(){{let rows=[['blind_id','image_path','technical_quality','hurts_listing_presentation','fixable_by_retake','defect_tags','notes']];
document.querySelectorAll('.card').forEach(x=>{{let b=x.dataset.bid;
let img=x.querySelector('img').getAttribute('src').replace('file://','');
let tags=[...x.querySelectorAll('input[name="tag_'+b+'"]:checked')].map(y=>y.value).join(';');
let notes=x.querySelector('input[name="notes_'+b+'"]').value;
rows.push([b,img,g(b,'tq'),g(b,'hurt'),g(b,'fix'),tags,notes].map(e))}});
let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.join(',')).join('\\n')],{{type:'text/csv'}}));
a.download='holdout_label_sheet_filled.csv';a.click()}}
st();
</script>"""
    path.write_text(html)


def _chatgpt_bundle(full, out: Path):
    from PIL import Image, ImageDraw, ImageFont
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)
    recs = []
    for _, r in full.iterrows():
        try:
            im = Image.open(r.image_path).convert("RGB")
        except Exception:
            continue
        im.save(out / "images" / f"{r.blind_id}.jpg", "JPEG", quality=88)
        recs.append((r.blind_id, r.image_path))
    CELL, BAR, COLS, ROWS, PAD, PER = 360, 28, 4, 4, 6, 16
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    sheets = math.ceil(len(recs) / PER)
    for s in range(sheets):
        batch = recs[s * PER:(s + 1) * PER]
        W = COLS * (CELL + PAD) + PAD
        H = ROWS * (CELL + BAR + PAD) + PAD
        sheet = Image.new("RGB", (W, H), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        for i, (bid, p) in enumerate(batch):
            cx = PAD + (i % COLS) * (CELL + PAD)
            cy = PAD + (i // COLS) * (CELL + BAR + PAD)
            d.rectangle([cx, cy, cx + CELL, cy + BAR], fill=(30, 30, 30))
            d.text((cx + 6, cy + 3), bid, fill=(255, 255, 255), font=font)
            im = Image.open(out / "images" / f"{bid}.jpg").convert("RGB")
            im.thumbnail((CELL, CELL))
            sheet.paste(im, (cx + (CELL - im.width) // 2, cy + BAR + (CELL - im.height) // 2))
        sheet.save(out / "sheets" / f"sheet_{s+1:02d}.jpg", "JPEG", quality=85)
    (out / "PROMPT.md").write_text(
        "# Holdout labeling — first-image technical photo quality only\n\n"
        "You are labeling marketplace listing photos for TECHNICAL PHOTO QUALITY ONLY. "
        "I upload contact sheets; each cell has an ID (e.g. H0001). Judge only the "
        "photograph (blur, dark, overexposed, glare hiding the item, bad crop, extreme "
        "tilt, low resolution/noise, clutter, item not clear, screenshot/not_item_photo). "
        "Ignore product value, price, brand.\n\n"
        "Return ONE CSV, header exactly:\n\n"
        "```\nblind_id,technical_quality,hurts_listing_presentation,fixable_by_retake,defect_tags,notes\n```\n\n"
        "- technical_quality: good | bad | uncertain | not_item_photo\n"
        "- hurts_listing_presentation: yes | no | uncertain\n"
        "- fixable_by_retake: yes | no | uncertain\n"
        "- defect_tags: semicolon list from "
        "blur;dark;overexposed;glare;bad_crop;extreme_tilt;low_resolution;noise;clutter;item_not_clear;other\n"
        "- one row per ID, keep uncertain explicit, avoid commas in notes.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scored_with_v4.csv")
    ap.add_argument("--exclude", default="data/evaluation/eval_candidates_private.csv")
    ap.add_argument("--per-method-search", type=int, default=3)
    ap.add_argument("--random-per-search", type=int, default=2)
    ap.add_argument("--repeat-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out-dir", default="data/holdout")
    build(ap.parse_args())
