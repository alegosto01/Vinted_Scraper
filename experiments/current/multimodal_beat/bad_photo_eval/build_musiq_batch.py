"""Confirmation batch to pin MUSIQ precision: sample by worst-MUSIQ across strata.

MUSIQ's fresh-holdout AUROC (0.85) is solid but top-K precision is noisy (only 8 fresh
bad photos). This draws a new labelable set stratified by musiq (worst tail + boundary +
random + a few best), excluding everything already labeled, so we can measure P@K for the
MUSIQ ranking on a clean set. Full technical_quality + defect_tags schema so we also learn
which defects MUSIQ catches. Deterministic per --seed.

Usage:
  python build_musiq_batch.py --scores data/scored_with_musiq.csv --n 60 --seed 20260727
"""
from __future__ import annotations
import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TQ = ["good", "bad", "uncertain", "not_item_photo"]
YN = ["yes", "no", "uncertain"]
TAGS = ["blur", "dark", "overexposed", "glare", "bad_crop", "extreme_tilt",
        "low_resolution", "noise", "clutter", "item_not_clear", "other"]
EXCLUDE = ["data/evaluation/eval_candidates_private.csv",
           "data/holdout/holdout_candidates_private.csv"]
EXCLUDE_GLOBS = ["data/blur_batch*/blur_candidates_private.csv",
                 "data/musiq_batch*/musiq_candidates_private.csv"]


def build(a):
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.scores, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)
    excl = set()
    for f in EXCLUDE + [g for gl in EXCLUDE_GLOBS for g in glob.glob(gl)]:
        if Path(f).exists():
            excl |= set(pd.read_csv(f, usecols=["item_id"])["item_id"].astype(str))
    pool = df[(df["image_available"] == 1) & df["image_path"].notna() &
              df["musiq"].notna() & (~df["item_id"].isin(excl))].copy()
    # worst = LOW musiq. Stratify on musiq quantiles.
    q = pool["musiq"].quantile
    strata = {
        "worst_musiq":  (pool[pool.musiq <= q(0.03)], round(a.n * 0.45)),
        "boundary":     (pool[(pool.musiq > q(0.03)) & (pool.musiq <= q(0.15))], round(a.n * 0.25)),
        "mid":          (pool[(pool.musiq > q(0.15)) & (pool.musiq <= q(0.6))], round(a.n * 0.15)),
        "good_control": (pool[pool.musiq >= q(0.9)], round(a.n * 0.15)),
    }
    picks = {}
    for name, (sub, k) in strata.items():
        if len(sub) and k > 0:
            for iid in sub.sample(min(k, len(sub)),
                                  random_state=int(rng.integers(1 << 31)))["item_id"]:
                picks.setdefault(iid, name)
    sel = df[df["item_id"].isin(picks)].drop_duplicates("item_id").copy()
    sel["stratum"] = sel["item_id"].map(picks)
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    sel["blind_id"] = [f"Q{i+1:04d}" for i in range(len(sel))]

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pcols = ["blind_id", "item_id", "search_names", "title", "image_path", "stratum",
             "musiq", "musiq_bad", "blur_score"]
    sel[[c for c in pcols if c in sel.columns]].to_csv(
        out / "musiq_candidates_private.csv", index=False)
    blind = sel[["blind_id", "image_path"]].copy()
    for c in ["technical_quality", "hurts_listing_presentation", "fixable_by_retake",
              "defect_tags", "notes"]:
        blind[c] = ""
    blind.to_csv(out / "musiq_label_sheet.csv", index=False)
    json.dump({"seed": a.seed, "n": int(len(sel)), "excluded": len(excl),
               "strata": {k: int((sel.stratum == k).sum()) for k in strata},
               "created_at": datetime.now(timezone.utc).isoformat()},
              open(out / "musiq_manifest.json", "w"), indent=2)
    _gallery(sel, out / "musiq_blind_gallery.html")
    _bundle(sel, out / "chatgpt_bundle")
    print(f"n={len(sel)} excluded={len(excl)} "
          f"strata={ {k:int((sel.stratum==k).sum()) for k in strata} } -> {out}")


def _gallery(sel, path):
    cards = []
    for _, r in sel.iterrows():
        b = r.blind_id
        tq = "".join(f'<label><input type=radio name="tq_{b}" value="{o}"> {o}</label>' for o in TQ)
        hurt = "".join(f'<label><input type=radio name="h_{b}" value="{o}"> {o}</label>' for o in YN)
        fix = "".join(f'<label><input type=radio name="f_{b}" value="{o}"> {o}</label>' for o in YN)
        tags = "".join(f'<label><input type=checkbox name="t_{b}" value="{t}"> {t}</label>' for t in TAGS)
        cards.append(f"""<div class=card data-bid="{b}">
  <img src="file://{r.image_path}" loading=lazy>
  <div class=f><b>{b}</b>
    <div class=row><span class=l>quality</span>{tq}</div>
    <div class=row><span class=l>hurts?</span>{hurt}</div>
    <div class=row><span class=l>fixable?</span>{fix}</div>
    <div class=row><span class=l>defects</span>{tags}</div>
    <div class=row><input class=notes name="n_{b}" placeholder="notes"></div>
  </div></div>""")
    html = f"""<!doctype html><meta charset=utf-8><title>MUSIQ confirmation labeling</title>
<style>body{{font:14px sans-serif;background:#111;color:#eee;margin:16px}}
.bar{{position:sticky;top:0;background:#000;padding:10px;margin:-16px -16px 10px;z-index:9}}
button{{padding:8px 14px;background:#2b7;border:0;border-radius:6px;color:#000;cursor:pointer}}
#p{{margin-left:12px;color:#9cf}}
.card{{display:inline-block;width:320px;vertical-align:top;margin:8px;background:#1c1c1c;border-radius:8px;padding:8px}}
.card img{{width:100%;border-radius:4px}} .f{{font-size:12px;margin-top:6px}} .row{{margin:3px 0}}
.l{{color:#888;margin-right:4px}} .f label{{margin-right:6px;white-space:nowrap;display:inline-block}}
.notes{{width:95%;background:#111;border:1px solid #444;color:#eee}} .card.done{{outline:2px solid #2b7}}</style>
<div class=bar><button onclick="dl()">Download CSV</button><span id=p></span>
<span style="color:#888;margin-left:12px">Judge ONLY first-image technical quality — not product/price.</span></div>
{''.join(cards)}
<script>
function st(){{let d=0,c=document.querySelectorAll('.card');c.forEach(x=>{{let b=x.dataset.bid;
if(x.querySelector('input[name="tq_'+b+'"]:checked')){{x.classList.add('done');d++}}else x.classList.remove('done')}});
document.getElementById('p').textContent=d+' / '+c.length+' done'}}
document.addEventListener('change',st);document.addEventListener('input',st);
function e(s){{s=(s==null?'':''+s);return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}}
function g(b,n){{let x=document.querySelector('input[name="'+n+'_'+b+'"]:checked');return x?x.value:''}}
function dl(){{let rows=[['blind_id','image_path','technical_quality','hurts_listing_presentation','fixable_by_retake','defect_tags','notes']];
document.querySelectorAll('.card').forEach(x=>{{let b=x.dataset.bid;
let img=x.querySelector('img').getAttribute('src').replace('file://','');
let tags=[...x.querySelectorAll('input[name="t_'+b+'"]:checked')].map(y=>y.value).join(';');
rows.push([b,img,g(b,'tq'),g(b,'h'),g(b,'f'),tags,x.querySelector('input[name="n_'+b+'"]').value].map(e))}});
let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.join(',')).join('\\n')],{{type:'text/csv'}}));
a.download='musiq_label_sheet_filled.csv';a.click()}}
st();
</script>"""
    path.write_text(html)


def _bundle(sel, out):
    from PIL import Image, ImageDraw, ImageFont
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "sheets").mkdir(exist_ok=True)
    recs = []
    for _, r in sel.iterrows():
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
    for s in range(math.ceil(len(recs) / PER)):
        batch = recs[s * PER:(s + 1) * PER]
        sheet = Image.new("RGB", (COLS * (CELL + PAD) + PAD, ROWS * (CELL + BAR + PAD) + PAD),
                          (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        for i, (bid, p) in enumerate(batch):
            cx = PAD + (i % COLS) * (CELL + PAD); cy = PAD + (i // COLS) * (CELL + BAR + PAD)
            d.rectangle([cx, cy, cx + CELL, cy + BAR], fill=(30, 30, 30))
            d.text((cx + 6, cy + 3), bid, fill=(255, 255, 255), font=font)
            im = Image.open(out / "images" / f"{bid}.jpg").convert("RGB"); im.thumbnail((CELL, CELL))
            sheet.paste(im, (cx + (CELL - im.width) // 2, cy + BAR + (CELL - im.height) // 2))
        sheet.save(out / "sheets" / f"sheet_{s+1:02d}.jpg", "JPEG", quality=85)
    (out / "PROMPT.md").write_text(
        "# Photo-quality labeling — first image only\n\nJudge ONLY technical photo quality "
        "(blur, dark, overexposed, glare, bad crop, extreme tilt, low resolution/noise, "
        "clutter, item not clear, screenshot=not_item_photo). Ignore product/price/brand.\n\n"
        "Return ONE CSV, header exactly:\n```\nblind_id,technical_quality,hurts_listing_presentation,"
        "fixable_by_retake,defect_tags,notes\n```\n- technical_quality: good|bad|uncertain|not_item_photo\n"
        "- hurts/fixable: yes|no|uncertain\n- defect_tags: semicolon list from "
        "blur;dark;overexposed;glare;bad_crop;extreme_tilt;low_resolution;noise;clutter;item_not_clear;other\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scored_with_musiq.csv")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--out-dir", default="data/musiq_batch")
    build(ap.parse_args())
