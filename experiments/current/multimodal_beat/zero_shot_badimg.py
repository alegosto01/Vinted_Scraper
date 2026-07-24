"""CLIP zero-shot bad-image detector for live listings.

Reuses the cached CLIP-image embeddings (img_clip_<tag>.npy from embed_blocks.py) and
scores each against good vs bad text prompts in CLIP space. No training, no re-encoding.
bad_score = softmax P(bad prompts) averaged. Ranks worst photos, copies them for review.

Usage: python zero_shot_badimg.py --csv <live_scored.csv> --emb-tag live19k --top 40
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_blocks import build_id_index, first_image  # noqa: E402
CACHE = Path(__file__).resolve().parent / "cache"

# Target TECHNICAL photo defects only: blurry / dark / misaligned. A person wearing the
# item (mirror or not) is GOOD -- it shows fit -- so it belongs in the GOOD set.
GOOD = ["a clear sharp in-focus photo", "a bright evenly well-lit photo",
        "a person wearing the clothing item", "a straight well-framed photo",
        "a normal quality product photo"]
BAD = ["a blurry out-of-focus photo", "a dark underexposed dim photo",
       "a crooked tilted misaligned photo", "a grainy low quality noisy photo",
       "a motion blurred photo"]


def clip_text_matrix(prompts):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        inp = p(text=prompts, return_tensors="pt", padding=True)
        t = m.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        v = m.text_projection(t.pooler_output).numpy()
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)).astype(np.float32), float(m.logit_scale.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb-tag", required=True)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--searches", default=None, help="comma list to filter SearchName")
    ap.add_argument("--out-dir", default=None, help="where to copy worst images (default: cache/badimg_<tag>)")
    a = ap.parse_args()

    img = np.load(CACHE / f"img_clip_{a.emb_tag}.npy")
    mask = np.load(CACHE / f"img_clip_{a.emb_tag}_mask.npy")
    df = pd.read_csv(a.csv, low_memory=False)
    assert len(df) == len(img), f"row mismatch {len(df)} vs {len(img)}"

    txt, scale = clip_text_matrix(GOOD + BAD)
    sims = img @ txt.T * scale                       # (N, nprompts) CLIP logits
    ng = len(GOOD)
    e = np.exp(sims - sims.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    bad_score = prob[:, ng:].sum(1)                  # P(any bad prompt)

    df = df.copy()
    df["bad_score"] = bad_score
    df["has_img"] = mask
    df = df[df["has_img"] == 1]
    if a.searches:
        keep = {s.strip() for s in a.searches.split(",")}
        df = df[df["SearchName"].astype(str).isin(keep)]
    df = df.sort_values("bad_score", ascending=False)

    idx = build_id_index()
    out_dir = Path(a.out_dir) if a.out_dir else CACHE / f"badimg_{a.emb_tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cols = [c for c in ["item_id", "SearchName", "Title", "Price", "Link", "bad_score"] if c in df.columns]
    top = df.head(a.top)
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        iid = str(r["item_id"])
        d = idx.get(iid)
        f = first_image(d) if d else None
        dst = ""
        if f:
            dst = str(out_dir / f"{rank:02d}_{r.get('SearchName','?')}_{iid}{Path(f).suffix}")
            shutil.copy(f, dst)
        rows.append({**{c: r[c] for c in cols}, "img": dst})
    review = pd.DataFrame(rows)
    review.to_csv(out_dir / "worst.csv", index=False)
    print(f"scored {len(df)} images | mean bad_score={df['bad_score'].mean():.3f}")
    print(f"worst {len(top)} copied to {out_dir}")
    print(review[[c for c in ["bad_score", "SearchName", "Price", "Title", "Link"] if c in review.columns]]
          .head(20).to_string(index=False, max_colwidth=45))


if __name__ == "__main__":
    main()
