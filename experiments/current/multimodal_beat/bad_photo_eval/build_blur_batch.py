"""Targeted blur-labeling batch: sample ACROSS the blur_score range.

Sampling only the blurriest tail would return mostly plain low-texture photos and few
clear positives; sampling only randomly would return almost no blur. So we stratify by
blur_score (very-high / moderate / low / random) to get a labelable mix of true blur,
sharp, plain, and controls. Excludes every item already labeled (the 244 dev set + the
holdout). Deterministic per --seed. Emits a blur-only blind gallery + ChatGPT bundle.

Usage:
  python build_blur_batch.py --scores data/scored_with_blur.csv --n 60 --seed 20260726
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

# exclude the dev/holdout sets AND every prior blur batch so batches never overlap
EXCLUDE_FILES = ["data/evaluation/eval_candidates_private.csv",
                 "data/holdout/holdout_candidates_private.csv"]
EXCLUDE_GLOBS = ["data/blur_batch*/blur_candidates_private.csv"]


def build(a):
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.scores, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)
    excl = set()
    files = list(EXCLUDE_FILES)
    for g in EXCLUDE_GLOBS:
        files += glob.glob(g)
    for f in files:
        p = Path(f)
        if p.exists():
            excl |= set(pd.read_csv(p, usecols=["item_id"])["item_id"].astype(str))
    pool = df[(df["image_available"] == 1) & df["image_path"].notna() &
              df["blur_score"].notna() & (~df["item_id"].isin(excl))].copy()

    q = pool["blur_score"].quantile
    if a.top_heavy:
        # round 2: chase positives (blur lives only in the top tail), keep a boundary
        # band to refine the threshold, minimal sharp/random.
        strata = {
            "very_blurry_candidate": (pool[pool.blur_score >= q(0.94)], round(a.n * 0.60)),
            "boundary":              (pool[(pool.blur_score >= q(0.85)) & (pool.blur_score < q(0.94))],
                                      round(a.n * 0.25)),
            "sharp":                 (pool[pool.blur_score <= q(0.30)], round(a.n * 0.08)),
            "random":                (pool, round(a.n * 0.07)),
        }
    else:
        strata = {
            "very_blurry_candidate": (pool[pool.blur_score >= q(0.97)], round(a.n * 0.42)),
            "moderate":              (pool[(pool.blur_score >= q(0.55)) & (pool.blur_score < q(0.97))],
                                      round(a.n * 0.25)),
            "sharp":                 (pool[pool.blur_score <= q(0.30)], round(a.n * 0.17)),
            "random":                (pool, round(a.n * 0.16)),
        }
    picks = {}
    for name, (sub, k) in strata.items():
        if len(sub) == 0 or k <= 0:
            continue
        take = sub.sample(min(k, len(sub)), random_state=int(rng.integers(1 << 31)))
        for iid in take["item_id"]:
            picks.setdefault(iid, name)

    sel = df[df["item_id"].isin(picks)].drop_duplicates("item_id").copy()
    sel["stratum"] = sel["item_id"].map(picks)
    order = rng.permutation(len(sel))
    sel = sel.iloc[order].reset_index(drop=True)
    sel["blind_id"] = [f"{a.prefix}{i+1:04d}" for i in range(len(sel))]

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    priv_cols = ["blind_id", "item_id", "search_names", "title", "image_path",
                 "stratum", "blur_score", "laplacian_variance", "edge_density",
                 "dynamic_range"]
    sel[[c for c in priv_cols if c in sel.columns]].to_csv(
        out / "blur_candidates_private.csv", index=False)
    blind = sel[["blind_id", "image_path"]].copy()
    blind["is_blurry"] = ""       # yes | no | uncertain
    blind["notes"] = ""
    blind.to_csv(out / "blur_label_sheet.csv", index=False)
    json.dump({"seed": a.seed, "n": int(len(sel)), "excluded": len(excl),
               "strata": {k: int((sel.stratum == k).sum()) for k in strata},
               "created_at": datetime.now(timezone.utc).isoformat()},
              open(out / "blur_manifest.json", "w"), indent=2)

    _gallery(sel, out / "blur_blind_gallery.html")
    _bundle(sel, out / "chatgpt_bundle")
    print(f"n={len(sel)} excluded={len(excl)} strata="
          f"{ {k:int((sel.stratum==k).sum()) for k in strata} } -> {out}")
    return sel


def _gallery(sel, path: Path):
    cards = []
    for _, r in sel.iterrows():
        b = r.blind_id
        radios = "".join(
            f'<label><input type=radio name="b_{b}" value="{o}"> {o}</label>'
            for o in ("yes", "no", "uncertain"))
        cards.append(f"""<div class=card data-bid="{b}">
  <img src="file://{r.image_path}" loading=lazy>
  <div class=f><b>{b}</b>
    <div class=row><span class=l>blurry / out of focus?</span>{radios}</div>
    <div class=row><input class=notes name="n_{b}" placeholder="notes"></div>
  </div></div>""")
    html = f"""<!doctype html><meta charset=utf-8><title>Blur labeling</title>
<style>body{{font:14px sans-serif;background:#111;color:#eee;margin:16px}}
.bar{{position:sticky;top:0;background:#000;padding:10px;margin:-16px -16px 10px;z-index:9}}
button{{padding:8px 14px;background:#2b7;border:0;border-radius:6px;color:#000;cursor:pointer}}
#p{{margin-left:12px;color:#9cf}}
.card{{display:inline-block;width:320px;vertical-align:top;margin:8px;background:#1c1c1c;
border-radius:8px;padding:8px}} .card img{{width:100%;border-radius:4px}}
.f{{font-size:13px;margin-top:6px}} .row{{margin:4px 0}} .l{{color:#888;margin-right:6px}}
.f label{{margin-right:8px}} .notes{{width:95%;background:#111;border:1px solid #444;color:#eee}}
.card.done{{outline:2px solid #2b7}}</style>
<div class=bar><button onclick="dl()">Download CSV</button><span id=p></span>
<span style="color:#888;margin-left:12px">Mark ONLY blur / out-of-focus. A sharp photo of a
plain item is NOT blurry. Ignore lighting, crop, clutter.</span></div>
{''.join(cards)}
<script>
function st(){{let d=0,c=document.querySelectorAll('.card');c.forEach(x=>{{
let b=x.dataset.bid;if(x.querySelector('input[name="b_'+b+'"]:checked')){{x.classList.add('done');d++}}else x.classList.remove('done')}});
document.getElementById('p').textContent=d+' / '+c.length+' done'}}
document.addEventListener('change',st);
function e(s){{s=(s==null?'':''+s);return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}}
function dl(){{let rows=[['blind_id','image_path','is_blurry','notes']];
document.querySelectorAll('.card').forEach(x=>{{let b=x.dataset.bid;
let img=x.querySelector('img').getAttribute('src').replace('file://','');
let v=x.querySelector('input[name="b_'+b+'"]:checked');
rows.push([b,img,v?v.value:'',x.querySelector('input[name="n_'+b+'"]').value].map(e))}});
let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.join(',')).join('\\n')],{{type:'text/csv'}}));
a.download='blur_label_sheet_filled.csv';a.click()}}
st();
</script>"""
    path.write_text(html)


def _bundle(sel, out: Path):
    from PIL import Image, ImageDraw, ImageFont
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)
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
        sheet = Image.new("RGB", (COLS * (CELL + PAD) + PAD,
                                  ROWS * (CELL + BAR + PAD) + PAD), (245, 245, 245))
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
        "# Blur labeling — is the first photo blurry / out of focus?\n\n"
        "I upload contact sheets; each cell has an ID (e.g. L0001). For each ID decide ONLY "
        "whether the photo is blurry or out of focus (soft, smeared, missed focus, motion "
        "blur). A SHARP photo of a plain/simple item is NOT blurry. Ignore lighting, crop, "
        "clutter, tilt — blur only.\n\n"
        "Return ONE CSV, header exactly:\n```\nblind_id,is_blurry,notes\n```\n"
        "- is_blurry: yes | no | uncertain\n- one row per ID; keep uncertain explicit; "
        "avoid commas in notes.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scored_with_blur.csv")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out-dir", default="data/blur_batch")
    ap.add_argument("--top-heavy", action="store_true")
    ap.add_argument("--prefix", default="L")
    build(ap.parse_args())
