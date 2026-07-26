"""Ship: rank listings by blur_score and surface the blurriest top-N per search.

blur_score (blur_score.py) is the one validated single-defect ranker (AUROC ~0.82,
P@5 1.0 on human labels). This produces a review queue: the N blurriest first-images
per search, as CSV + an HTML gallery. Ranking only — not a calibrated probability.

Usage:
  python rank_blur.py --scores data/scored_with_blur.csv --top 15 --out-dir data/blur_ranking
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from blur_score import add_blur_score

VINTED = "https://www.vinted.it/items/"


def rank(scores: Path, top: int, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(scores, low_memory=False)
    if "blur_score" not in t.columns:
        t = add_blur_score(t)
    t = t[t["blur_score"].notna()].copy()
    t["item_id"] = t["item_id"].astype(str)
    t["link"] = VINTED + t["item_id"]
    t["search_list"] = t["search_names"].fillna("").astype(str).apply(lambda s: s.split("|"))

    searches = sorted({s for lst in t["search_list"] for s in lst if s})
    frames = []
    for s in searches:
        sub = t[t["search_list"].apply(lambda l: s in l)].nlargest(top, "blur_score")
        sub = sub.assign(search=s, blur_rank=range(1, len(sub) + 1))
        frames.append(sub)
    ranked = pd.concat(frames, ignore_index=True)
    cols = ["search", "blur_rank", "item_id", "blur_score", "title", "link", "image_path"]
    ranked[cols].to_csv(out / "blur_ranking.csv", index=False)
    _gallery(ranked, searches, top, out / "blur_ranking_gallery.html")
    print(f"wrote {out}/blur_ranking.csv ({len(ranked)} rows, "
          f"{len(searches)} searches x top {top})")
    return ranked


def _gallery(ranked, searches, top, path: Path):
    blocks = []
    for s in searches:
        sub = ranked[ranked["search"] == s]
        cards = "".join(f"""<div class=card>
  <img src="file://{r.image_path}" loading=lazy>
  <div class=m>#{int(r.blur_rank)} · blur={r.blur_score:.2f}<br>
  <a href="{r.link}" target=_blank>{str(r.title)[:50]}</a></div></div>"""
                        for _, r in sub.iterrows())
        blocks.append(f"<h2>{s}</h2><div class=grid>{cards}</div>")
    html = f"""<!doctype html><meta charset=utf-8><title>Blurriest listings</title>
<style>body{{font:14px sans-serif;background:#111;color:#eee;margin:16px}}
h2{{color:#9cf;border-bottom:1px solid #333;padding-top:12px}}
.grid{{display:flex;flex-wrap:wrap}}
.card{{width:220px;margin:6px;background:#1c1c1c;border-radius:6px;padding:6px}}
.card img{{width:100%;border-radius:4px}} .m{{font-size:12px;margin-top:4px}}
a{{color:#7cf}}</style>
<h1>Blurriest first-images — top {top} per search</h1>
<p>Ranking by blur_score (validated blur detector). Review candidates; not a probability.</p>
{''.join(blocks)}"""
    path.write_text(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scored_with_blur.csv")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out-dir", default="data/blur_ranking")
    a = ap.parse_args()
    rank(Path(a.scores), a.top, Path(a.out_dir))
