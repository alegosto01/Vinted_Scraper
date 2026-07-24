"""Precompute & cache frozen embedding blocks for the multimodal experiment.

Encoders (all cached locally, CPU, frozen):
  image: CLIP-ViT-B/32 (openai/clip-vit-base-patch32) -> 512d, from image_cache/<item_id>/*.webp
  text : minilm-multi (done elsewhere), bert-base-uncased, clip-text, mDeBERTa

Images resolved by item_id across all image_cache dirs. Missing image -> zeros + mask=0.
Outputs npy blocks under cache/, aligned to the CSV row order.
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/ale/Desktop/vinted/Vinted_New_Version")
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def build_id_index() -> dict[str, str]:
    """Map item_id -> image dir across all caches. Flat caches are keyed by <item_id>;
    the collector's live_runs caches are nested image_cache/<search>/<item_id>."""
    idx = {}
    flat = glob.glob(str(ROOT / "data/simple_scrape/*/image_cache")) + \
        glob.glob(str(ROOT / "experiments/current/time_to_sell/data/fast_discovery_images/image_cache"))
    for c in flat:
        try:
            for d in os.listdir(c):
                idx.setdefault(d, os.path.join(c, d))
        except OSError:
            continue
    for search_dir in glob.glob(str(ROOT / "experiments/current/time_to_sell/data/live_runs/bin_collector_*/image_cache/*")):
        try:
            for d in os.listdir(search_dir):
                idx.setdefault(d, os.path.join(search_dir, d))
        except OSError:
            continue
    return idx


def first_image(item_dir: str):
    for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png"):
        g = sorted(Path(item_dir).glob(ext))
        if g:
            return g[0]
    return None


def clip_image_embed(ids, tag, limit=None):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    out_path = CACHE / f"img_clip_{tag}.npy"
    mask_path = CACHE / f"img_clip_{tag}_mask.npy"
    if out_path.exists() and limit is None:
        return np.load(out_path), np.load(mask_path)
    idx = build_id_index()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    dim = model.config.projection_dim
    ids = ids[:limit] if limit else ids
    emb = np.zeros((len(ids), dim), dtype=np.float32)
    mask = np.zeros(len(ids), dtype=np.float32)
    with torch.no_grad():
        for i, iid in enumerate(ids):
            d = idx.get(str(iid))
            f = first_image(d) if d else None
            if not f:
                continue
            try:
                img = Image.open(f).convert("RGB")
                inp = proc(images=img, return_tensors="pt")
                vout = model.vision_model(pixel_values=inp["pixel_values"])
                v = model.visual_projection(vout.pooler_output)[0].numpy()
                emb[i] = v / (np.linalg.norm(v) + 1e-8)
                mask[i] = 1.0
            except Exception as e:
                if limit is not None:
                    print("  img err:", type(e).__name__, e)
                continue
            if limit is None and (i + 1) % 100 == 0:
                print(f"  clip-img {i+1}/{len(ids)} (hit={int(mask.sum())})", flush=True)
    if limit is None:
        np.save(out_path, emb); np.save(mask_path, mask)
    print(f"clip-image: {int(mask.sum())}/{len(ids)} images embedded")
    return emb, mask


def clip_text_embed(titles, tag):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    out = CACHE / f"txt_clip_{tag}.npy"
    if out.exists():
        return np.load(out)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    embs = []
    with torch.no_grad():
        for i in range(0, len(titles), 128):
            batch = [t[:77] or " " for t in titles[i:i + 128]]
            inp = proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
            tout = model.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            v = model.text_projection(tout.pooler_output).numpy()
            embs.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
    emb = np.vstack(embs).astype(np.float32)
    np.save(out, emb)
    print(f"clip-text: {emb.shape}")
    return emb


def hf_text_embed(titles, model_name, block, tag):
    import torch
    from transformers import AutoModel, AutoTokenizer
    out = CACHE / f"{block}_{tag}.npy"
    if out.exists():
        return np.load(out)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(titles), 64):
            batch = [t or " " for t in titles[i:i + 64]]
            inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=32)
            hs = model(**inp).last_hidden_state
            m = inp["attention_mask"].unsqueeze(-1).float()
            v = (hs * m).sum(1) / m.sum(1).clamp(min=1)   # mean pooling
            v = v.numpy()
            embs.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
    emb = np.vstack(embs).astype(np.float32)
    np.save(out, emb)
    print(f"{block}: {emb.shape}")
    return emb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="compute image + all text encoders")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, low_memory=False)
    ids = df["item_id"].astype(str).tolist() if "item_id" in df else df["Dataid"].astype(str).tolist()
    titles = df["Title"].fillna("").astype(str).tolist()
    if a.all:
        clip_text_embed(titles, a.tag)
        hf_text_embed(titles, "bert-base-uncased", "txt_bert", a.tag)
        hf_text_embed(titles, "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "txt_mdeberta", a.tag)
        emb, mask = clip_image_embed(ids, a.tag, a.limit)
        print("img coverage", float(mask.mean()))
    else:
        emb, mask = clip_image_embed(ids, a.tag, a.limit)
        print("shape", emb.shape, "coverage", float(mask.mean()))
