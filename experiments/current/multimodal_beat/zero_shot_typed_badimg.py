"""Type-aware CLIP zero-shot bad-image detector (v2).

Stage 1 (typing): decide item TYPE from SEARCH + TITLE keywords (reliable — the brand
  searches gucci/prada/griffati mix perfume/clothing/bags/shoes/eyewear). Image-CLIP
  detection is only a fallback when keywords are inconclusive.
Stage 2 (quality): score photo quality with GOOD-vs-BAD prompts SPECIFIC to that type,
  then RANK WITHIN each type (absolute P(bad) is not comparable across types).

Reuses cached CLIP-image embeddings (embed_blocks.py). Copies worst-N-per-type images
into a temp folder, subfoldered by type.
"""
from __future__ import annotations
import argparse, re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_blocks import build_id_index, first_image  # noqa: E402
CACHE = Path(__file__).resolve().parent / "cache"

# ---- type-specific quality prompts (good vs bad on OPTICS, item type held constant) ----
TYPES = {
    "video_game": {
        "detect": ["a video game case, box art, or game disc"],
        "good": ["a sharp, bright, straight-on photo of a video game case with a clearly readable cover"],
        "bad": ["a blurry, out-of-focus, dark, or glare-covered photo of a video game case"],
    },
    "console_device": {
        "detect": ["a game console, controller, steering wheel, or gaming hardware"],
        "good": ["a sharp, bright, well-framed photo of gaming hardware on a clean surface"],
        "bad": ["a dark, blurry, or cluttered photo of gaming hardware on a messy floor"],
    },
    "phone_tablet": {
        "detect": ["a smartphone or tablet"],
        "good": ["a sharp, bright photo clearly showing a phone or tablet"],
        "bad": ["a dark, blurry, or reflection-covered photo of a phone or tablet"],
    },
    "shoes": {
        "detect": ["a pair of shoes or sneakers"],
        "good": ["a sharp, bright, well-framed photo of shoes"],
        "bad": ["a blurry, dark, or messy photo of shoes on a cluttered floor"],
    },
    "clothing": {
        "detect": ["a piece of clothing or apparel"],
        "good": ["a sharp, bright, well-framed photo of clothing on a hanger, flat lay, or worn"],
        "bad": ["a blurry, dark, crumpled, or badly framed photo of clothing"],
    },
    "bag": {
        "detect": ["a handbag, purse, or backpack"],
        "good": ["a sharp, bright, well-framed photo of a bag"],
        "bad": ["a blurry, dark, or cluttered photo of a bag"],
    },
    "perfume": {
        "detect": ["a perfume or cologne bottle"],
        "good": ["a sharp, bright, well-framed photo of a perfume bottle centered on a clean surface"],
        "bad": ["a blurry, out-of-focus, dark, or badly framed photo of a perfume bottle"],
    },
    "jewelry": {
        "detect": ["jewelry: a necklace, ring, bracelet, or earrings"],
        "good": ["a sharp, bright, in-focus close-up photo of jewelry"],
        "bad": ["a blurry, out-of-focus, or dark photo of jewelry"],
    },
    "eyewear": {
        "detect": ["eyeglasses or sunglasses"],
        "good": ["a sharp, bright, in-focus photo of eyeglasses or sunglasses"],
        "bad": ["a blurry, dark, or reflection-covered photo of eyeglasses"],
    },
    "collectible": {
        "detect": ["trading cards, collectible cards, or a small collectible"],
        "good": ["a sharp, bright, in-focus, readable photo of trading cards or a collectible"],
        "bad": ["a blurry, out-of-focus, dark, or glare-covered photo of trading cards"],
    },
}

# ---- title keyword rules (multilingual it/fr/en/de); first match wins, order matters ----
TITLE_RULES = [
    ("perfume", r"parfum|profumo|eau de|fragrance|cologne|\bedp\b|\bedt\b|\bml\b"),
    ("eyewear", r"occhial|sunglass|lunettes|brille|\bglasses\b|eyewear"),
    ("bag", r"borsa|borsell|\bsac\b|\bbag\b|purse|zaino|backpack|pochette|clutch|tasche|handtasche"),
    ("shoes", r"scarp|sneaker|\bshoes?\b|chaussure|schuhe|air max|air force|jordan|stivali|boot|sandal|baskets|tenis"),
    ("jewelry", r"collan|collier|necklace|anello|\bring\b|\bbague\b|braccial|bracelet|orecchin|earring|\bkette\b|gioiell"),
    ("phone_tablet", r"iphone|samsung|galaxy|xiaomi|huawei|pixel|telefon|smartphone|\btablet\b|ipad|\bcellulare\b"),
    ("console_device", r"volant|steering wheel|\bconsole\b|controller|\bpad\b|joystick|casque|cuffie|headset|\bhori\b|\bkit\b"),
    ("video_game", r"\bgioco\b|\bgiochi\b|\bgame\b|\bjeu\b|\bjuego\b|\bps4\b|\bps5\b|blister|scellé|sigillat|precintad"),
    ("collectible", r"pok[eé]mon|carte|cards?|figurine|funko|collezion|amiibo|lego"),
]
# per-search fallback type when title has no keyword
SEARCH_DEFAULT = {
    "telefoni": "phone_tablet", "console": "console_device", "tablets": "phone_tablet",
    "donna_accessori_gioielli": "jewelry", "hobby_collezionismo": "collectible",
    "ps4": "video_game", "nike": "clothing", "gucci": "clothing", "prada": "clothing",
    "griffati_donna_all": "clothing", "griffati_uomo_all": "clothing",
}


def classify(search: str, title: str) -> str | None:
    t = str(title).lower()
    for typ, pat in TITLE_RULES:
        if re.search(pat, t):
            return typ
    return SEARCH_DEFAULT.get(str(search))


def clip_text(prompts):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        inp = p(text=prompts, return_tensors="pt", padding=True)
        tt = m.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        v = m.text_projection(tt.pooler_output).numpy()
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return v.astype(np.float32), float(m.logit_scale.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb-tag", required=True)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--searches", default=None)
    a = ap.parse_args()

    img = np.load(CACHE / f"img_clip_{a.emb_tag}.npy")
    mask = np.load(CACHE / f"img_clip_{a.emb_tag}_mask.npy")
    df = pd.read_csv(a.csv, low_memory=False)
    names = list(TYPES)

    # ---- stage 1: type from search+title, image-CLIP as fallback ----
    itype = np.array([classify(s, t) for s, t in zip(df.get("SearchName", ""), df.get("Title", ""))],
                     dtype=object)
    need = np.where(pd.isna(itype) | (itype == None))[0]  # noqa: E711
    if len(need):
        det_prompts = [TYPES[t]["detect"][0] for t in names]
        det, _ = clip_text(det_prompts)
        guess = np.array(names)[(img[need] @ det.T).argmax(1)]
        for j, i in enumerate(need):
            itype[i] = guess[j]

    # ---- stage 2: per-type good/bad quality, rank within type ----
    gb, owner, kind = [], [], []
    for t in names:
        gb.append(TYPES[t]["good"][0]); owner.append(t); kind.append("good")
        gb.append(TYPES[t]["bad"][0]); owner.append(t); kind.append("bad")
    gbmat, scale = clip_text(gb)
    owner = np.array(owner); kind = np.array(kind)
    gbsim = img @ gbmat.T * scale
    bad_score = np.zeros(len(img), dtype=np.float32)
    for t in names:
        rows = np.where(itype == t)[0]
        if not len(rows):
            continue
        cg = np.where((owner == t) & (kind == "good"))[0]
        cb = np.where((owner == t) & (kind == "bad"))[0]
        lg = gbsim[np.ix_(rows, cg)].mean(1)
        lb = gbsim[np.ix_(rows, cb)].mean(1)
        bad_score[rows] = 1.0 / (1.0 + np.exp(lg - lb))

    df = df.copy()
    df["item_type"] = itype
    df["bad_score"] = bad_score
    df = df[mask == 1]
    if a.searches:
        keep = {s.strip() for s in a.searches.split(",")}
        df = df[df["SearchName"].astype(str).isin(keep)]
    df = df.sort_values("bad_score", ascending=False)

    idx = build_id_index()
    out = Path(a.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cols = [c for c in ["item_id", "SearchName", "item_type", "Title", "Price", "Link", "bad_score"] if c in df.columns]
    rows_out = []
    for t in names:
        sub_df = df[df["item_type"] == t].head(a.per_type)
        if sub_df.empty:
            continue
        sub = out / t
        sub.mkdir(exist_ok=True)
        for rank, (_, r) in enumerate(sub_df.iterrows(), 1):
            iid = str(r["item_id"]); d = idx.get(iid); f = first_image(d) if d else None
            dst = ""
            if f:
                dst = str(sub / f"{rank:02d}_{iid}_{r.get('SearchName','?')}{Path(f).suffix}")
                shutil.copy(f, dst)
            rows_out.append({**{c: r[c] for c in cols}, "img": dst})
    pd.DataFrame(rows_out).to_csv(out / "worst.csv", index=False)

    print("type distribution (search+title typing):")
    print(df["item_type"].value_counts().to_string())
    print(f"\nworst {a.per_type} PER TYPE -> {out}")
    for t in names:
        s = df[df["item_type"] == t]["bad_score"]
        if len(s):
            print(f"  {t:15s} n={len(s):5d} worst={s.max():.3f} median={s.median():.3f}")


if __name__ == "__main__":
    main()
