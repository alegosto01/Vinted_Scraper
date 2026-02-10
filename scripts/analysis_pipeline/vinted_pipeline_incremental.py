#!/usr/bin/env python3
"""
vinted_pipeline_incremental.py

Incremental mode:
- loads SQLite index (products + variants)
- reads ONLY new items (either from a small "new_items.csv" or from a growing CSV with last_seen_rows)
- assigns ProductId + VariantId (stable)
- updates centroids + variant price buffers
- computes DealScore for new items only
- appends results to an output CSV (or writes a new one)

Install:
  python -m pip install numpy pandas sentence-transformers

Run (best): pass only the new items you scraped this round
  python vinted_pipeline_incremental.py --new_items new_items.csv --db ./out/index.sqlite --out ./out/stream_assigned.csv

Run (ok): use a growing CSV and process only appended rows
  python vinted_pipeline_incremental.py --growing_csv big.csv --db ./out/index.sqlite --out ./out/stream_assigned.csv
"""

import argparse
import os
import re
import math
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

import analysis_pipeline.vinted_index_score as store

# --- same text normalization as batch ---
ALIAS_MAP = {
    r"\baf\s?1\b": "air force 1",
    r"\baf1\b": "air force 1",
    r"\blv\b": "louis vuitton",
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_text(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).lower().strip()
    s = strip_accents(s)
    for pat, repl in ALIAS_MAP.items():
        s = re.sub(pat, repl, s)
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    toks = TOKEN_RE.findall(s)
    return [t for t in toks if len(t) > 1]

def core_tokens_from_counts(counts, n_listings: int, min_frac: float):
    return {t for t, c in counts.items() if (c / max(1, n_listings)) >= min_frac}

def make_variant_text(title_norm: str, core: set):
    toks = tokenize(title_norm)
    return " ".join([t for t in toks if t not in core])

def robust_stats(x: np.ndarray):
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(med), float(mad)

def deal_score_variant(prices: np.ndarray, price: float):
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 3 or not np.isfinite(price) or price <= 0:
        return np.nan
    med = np.median(prices)
    mad = np.median(np.abs(prices - med))
    scale = 1.4826 * mad + 1e-6
    return float((med - price) / scale)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# --- embeddings ---
def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    def embed(texts):
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    return embed

# --- nearest with cosine distance (vectors are normalized) ---
def nearest_centroid(centroids: np.ndarray, x: np.ndarray):
    # cosine similarity = dot since normalized
    sims = centroids @ x
    j = int(np.argmax(sims))
    dist = float(1.0 - sims[j])
    return j, dist

def online_mean_update(mu: np.ndarray, n: int, x: np.ndarray):
    # new_mu = mu + (x - mu) / (n+1)
    return mu + (x - mu) / float(n + 1)

def top_tokens(texts, k=8):
    toks = []
    for s in texts:
        toks.extend(tokenize(s))
    return " ".join([w for w, _ in Counter(toks).most_common(k)])



def process_new_df(df_new: pd.DataFrame, db_path: str, price_buffer_size: int = 200):
    """
    Same logic as CLI, but takes a DataFrame directly and returns df_assigned.
    """
    con = store.connect(db_path)
    model_name = store.get_meta(con, "model_name", "paraphrase-multilingual-MiniLM-L12-v2")
    embed = get_embedder(model_name)

    products = store.load_products(con)
    variants = store.load_variants(con)
    next_pid, next_vid = store.get_next_ids(con)

    if df_new.empty:
        return df_new

    df_new = df_new.copy()
    df_new["Title_norm"] = df_new["Title"].apply(normalize_text)
    df_new["Brand_norm"] = df_new["Brand"].apply(normalize_text) if "Brand" in df_new.columns else ""
    size_col = df_new["Size"].astype(str) if "Size" in df_new.columns else ""
    df_new["EmbedText"] = (df_new["Title_norm"].fillna("") + " | " + df_new["Brand_norm"].fillna("") + " | " + size_col.fillna("")).astype(str)

    if not products:
        raise RuntimeError("Index has no products. Run batch first.")

    pids = sorted(products.keys())
    P = np.stack([products[pid]["centroid"] for pid in pids], axis=0).astype(np.float32)
    X = embed(df_new["EmbedText"].tolist())

    assigned_pids, assigned_vids, variant_texts, deal_scores, deal_notes = [], [], [], [], []

    for r_i, row in df_new.iterrows():
        x = X[df_new.index.get_loc(r_i)]
        j, dist = nearest_centroid(P, x)
        pid = pids[j]
        thr = products[pid]["product_threshold"]

        if dist > thr:
            assigned_pids.append(-1); assigned_vids.append(-1)
            variant_texts.append(""); deal_scores.append(np.nan); deal_notes.append("unassigned_product")
            continue

        # update product centroid online
        mu = products[pid]["centroid"]; n = products[pid]["n"]
        new_mu = online_mean_update(mu, n, x)
        products[pid]["centroid"] = new_mu
        products[pid]["n"] = n + 1
        store.upsert_product(
            con, pid, new_mu, products[pid]["n"], thr,
            products[pid]["canonical_name"], products[pid]["block_key_hint"],
            products[pid]["core_token_counts"], products[pid]["core_token_min_frac"]
        )

        assigned_pids.append(pid)

        # variant assignment
        core_counts = products[pid]["core_token_counts"]
        core_frac = products[pid]["core_token_min_frac"]
        core = core_tokens_from_counts(core_counts, products[pid]["n"], core_frac)

        vtext = make_variant_text(row["Title_norm"], core)
        variant_texts.append(vtext)

        vemb = embed([vtext])[0]
        price = pd.to_numeric(row.get("Price", np.nan), errors="coerce")
        price = float(price) if np.isfinite(price) else np.nan

        vdict = variants.get(pid, {})
        if not vdict:
            vid = next_vid; next_vid += 1
            price_dim = 0.0
            Vc = np.hstack([vemb, [price_dim]]).astype(np.float32)
            variants.setdefault(pid, {})[vid] = {
                "variant_id": vid, "product_id": pid, "centroid": Vc, "n": 1,
                "variant_threshold": 0.33, "price_weight": 0.35, "core_frac": core_frac,
                "price_buffer": [price] if np.isfinite(price) else [],
                "variant_text_top": top_tokens([vtext]),
            }
            store.upsert_variant(con, vid, pid, Vc, 1, 0.33, 0.35, core_frac,
                                 variants[pid][vid]["price_buffer"], variants[pid][vid]["variant_text_top"])
            assigned_vids.append(vid); deal_scores.append(np.nan); deal_notes.append("new_variant_created")
            continue

        vids = sorted(vdict.keys())
        Vmat = np.stack([vdict[v]["centroid"] for v in vids], axis=0).astype(np.float32)
        vthr = vdict[vids[0]]["variant_threshold"]
        pwt = vdict[vids[0]]["price_weight"]

        if np.isfinite(price) and price > 0:
            price_dim = float(pwt * np.log(price))
        else:
            price_dim = 0.0

        xv = np.hstack([vemb, [price_dim]]).astype(np.float32)
        k, vdist = nearest_centroid(Vmat, xv)
        chosen_vid = vids[k]

        if vdist > vthr:
            chosen_vid = next_vid; next_vid += 1
            Vc = xv.copy()
            vdict[chosen_vid] = {
                "variant_id": chosen_vid, "product_id": pid, "centroid": Vc, "n": 1,
                "variant_threshold": vthr, "price_weight": pwt, "core_frac": core_frac,
                "price_buffer": [price] if np.isfinite(price) else [],
                "variant_text_top": top_tokens([vtext]),
            }
            store.upsert_variant(con, chosen_vid, pid, Vc, 1, vthr, pwt, core_frac,
                                 vdict[chosen_vid]["price_buffer"], vdict[chosen_vid]["variant_text_top"])
            assigned_vids.append(chosen_vid); deal_scores.append(np.nan); deal_notes.append("new_variant_created")
            continue

        # update existing variant
        vinfo = vdict[chosen_vid]
        vinfo["centroid"] = online_mean_update(vinfo["centroid"], vinfo["n"], xv)
        vinfo["n"] += 1

        if np.isfinite(price) and price > 0:
            vinfo["price_buffer"].append(price)
            if len(vinfo["price_buffer"]) > price_buffer_size:
                vinfo["price_buffer"] = vinfo["price_buffer"][-price_buffer_size:]

        vinfo["variant_text_top"] = top_tokens([vinfo["variant_text_top"], vtext])

        store.upsert_variant(con, chosen_vid, pid, vinfo["centroid"], vinfo["n"],
                             vinfo["variant_threshold"], vinfo["price_weight"], vinfo["core_frac"],
                             vinfo["price_buffer"], vinfo["variant_text_top"])

        assigned_vids.append(chosen_vid)

        arr = np.array(vinfo["price_buffer"], dtype=np.float32)
        z = deal_score_variant(arr, price) if np.isfinite(price) else np.nan
        penalty = 0.0
        nn = []

        v_med, _ = robust_stats(arr)
        if np.isfinite(price) and np.isfinite(v_med) and v_med > 0 and price < 0.45 * v_med:
            penalty += 0.5; nn.append("very_low_vs_variant_median")

        score = clamp(z - penalty, -10, 10) if np.isfinite(z) else np.nan
        deal_scores.append(score)
        deal_notes.append("|".join(nn))

    df_new["ProductId"] = assigned_pids
    df_new["VariantId"] = assigned_vids
    df_new["VariantText"] = variant_texts
    df_new["DealScore"] = deal_scores
    df_new["DealNotes"] = deal_notes
    return df_new


def run(args):
    con = store.connect(args.db)
    model_name = store.get_meta(con, "model_name", "paraphrase-multilingual-MiniLM-L12-v2")
    embed = get_embedder(model_name)

    products = store.load_products(con)   # pid -> info
    variants = store.load_variants(con)   # pid -> {vid -> info}
    next_pid, next_vid = store.get_next_ids(con)

    # Load new items
    if args.new_items:
        df_new = pd.read_csv(args.new_items)
    else:
        # growing CSV mode: only process appended rows
        last_seen = store.get_processed_state(con, args.growing_csv)
        df_all = pd.read_csv(args.growing_csv)
        df_new = df_all.iloc[last_seen:].copy()
        store.update_processed_state(con, args.growing_csv, len(df_all))

    if df_new.empty:
        print("No new rows to process.")
        return

    # normalize
    df_new["Title_norm"] = df_new["Title"].apply(normalize_text)
    df_new["Brand_norm"] = df_new["Brand"].apply(normalize_text) if "Brand" in df_new.columns else ""
    size_col = df_new["Size"].astype(str) if "Size" in df_new.columns else ""
    df_new["EmbedText"] = (df_new["Title_norm"].fillna("") + " | " + df_new["Brand_norm"].fillna("") + " | " + size_col.fillna("")).astype(str)

    # Prepare product centroid matrix
    if not products:
        raise RuntimeError("Index has no products. Run batch first to build the index DB.")
    pids = sorted(products.keys())
    P = np.stack([products[pid]["centroid"] for pid in pids], axis=0).astype(np.float32)

    # Embed new item texts
    X = embed(df_new["EmbedText"].tolist())

    assigned_pids = []
    assigned_vids = []
    variant_texts = []
    deal_scores = []
    deal_notes = []

    for i, row in df_new.iterrows():
        x = X[df_new.index.get_loc(i)]

        # --- assign product ---
        j, dist = nearest_centroid(P, x)
        pid = pids[j]
        thr = products[pid]["product_threshold"]

        if dist > thr:
            # out-of-distribution: buffer as "unassigned product"
            # you can either create a new ProductId here, or store for later batch refresh.
            # safer: keep unassigned
            assigned_pids.append(-1)
            assigned_vids.append(-1)
            variant_texts.append("")
            deal_scores.append(np.nan)
            deal_notes.append("unassigned_product")
            continue

        assigned_pids.append(pid)

        # update product centroid online
        mu = products[pid]["centroid"]
        n = products[pid]["n"]
        new_mu = online_mean_update(mu, n, x)
        products[pid]["centroid"] = new_mu
        products[pid]["n"] = n + 1
        store.upsert_product(
            con, pid, new_mu, products[pid]["n"], thr,
            products[pid]["canonical_name"], products[pid]["block_key_hint"],
            products[pid]["core_token_counts"], products[pid]["core_token_min_frac"]
        )

        # --- variant assignment within product ---
        core_counts = products[pid]["core_token_counts"]
        core_frac = products[pid]["core_token_min_frac"]
        core = core_tokens_from_counts(core_counts, products[pid]["n"], core_frac)

        vtext = make_variant_text(row["Title_norm"], core)
        variant_texts.append(vtext)

        vemb = embed([vtext])[0]  # normalized
        price = pd.to_numeric(row.get("Price", np.nan), errors="coerce")
        price = float(price) if np.isfinite(price) else np.nan

        # get variants for pid
        vdict = variants.get(pid, {})
        if not vdict:
            # create first variant for this product (rare if batch created properly)
            vid = next_vid; next_vid += 1
            # centroid in variant space = [vemb..., price_dim]
            price_dim = 0.0
            Vc = np.hstack([vemb, [price_dim]]).astype(np.float32)
            variants.setdefault(pid, {})[vid] = {
                "variant_id": vid, "product_id": pid, "centroid": Vc, "n": 1,
                "variant_threshold": 0.33, "price_weight": 0.35, "core_frac": core_frac,
                "price_buffer": [price] if np.isfinite(price) else [],
                "variant_text_top": top_tokens([vtext]),
            }
            store.upsert_variant(con, vid, pid, Vc, 1, 0.33, 0.35, core_frac, variants[pid][vid]["price_buffer"], variants[pid][vid]["variant_text_top"])
            assigned_vids.append(vid)
            deal_scores.append(np.nan)
            deal_notes.append("new_variant_created")
            continue

        # build centroid matrix for variants of pid
        vids = sorted(vdict.keys())
        Vmat = np.stack([vdict[v]["centroid"] for v in vids], axis=0).astype(np.float32)
        # build x in same variant space
        vthr = vdict[vids[0]]["variant_threshold"]
        pwt = vdict[vids[0]]["price_weight"]

        if np.isfinite(price) and price > 0:
            lp = np.log(price)
            # we don't have global mean/std in streaming; price dim still helps as relative signal
            price_dim = float(pwt * lp)
        else:
            price_dim = 0.0

        xv = np.hstack([vemb, [price_dim]]).astype(np.float32)

        k, vdist = nearest_centroid(Vmat, xv)
        chosen_vid = vids[k]
        if vdist > vthr:
            # either create new variant or buffer. I recommend create variant *within product*.
            chosen_vid = next_vid; next_vid += 1
            Vc = xv.copy()
            vdict[chosen_vid] = {
                "variant_id": chosen_vid, "product_id": pid, "centroid": Vc, "n": 1,
                "variant_threshold": vthr, "price_weight": pwt, "core_frac": core_frac,
                "price_buffer": [price] if np.isfinite(price) else [],
                "variant_text_top": top_tokens([vtext]),
            }
            store.upsert_variant(con, chosen_vid, pid, Vc, 1, vthr, pwt, core_frac, vdict[chosen_vid]["price_buffer"], vdict[chosen_vid]["variant_text_top"])
            assigned_vids.append(chosen_vid)
            deal_scores.append(np.nan)
            deal_notes.append("new_variant_created")
            continue

        # assign existing variant
        assigned_vids.append(chosen_vid)

        # update variant centroid online
        vinfo = vdict[chosen_vid]
        vinfo["centroid"] = online_mean_update(vinfo["centroid"], vinfo["n"], xv)
        vinfo["n"] += 1

        # update rolling price buffer
        if np.isfinite(price) and price > 0:
            vinfo["price_buffer"].append(price)
            if len(vinfo["price_buffer"]) > args.price_buffer_size:
                vinfo["price_buffer"] = vinfo["price_buffer"][-args.price_buffer_size:]

        # update top tokens cheaply
        vinfo["variant_text_top"] = top_tokens([vinfo["variant_text_top"], vtext])

        store.upsert_variant(
            con, chosen_vid, pid, vinfo["centroid"], vinfo["n"],
            vinfo["variant_threshold"], vinfo["price_weight"], vinfo["core_frac"],
            vinfo["price_buffer"], vinfo["variant_text_top"]
        )

        # deal score for this new item vs current price buffer
        arr = np.array(vinfo["price_buffer"], dtype=np.float32)
        z = deal_score_variant(arr, price) if np.isfinite(price) else np.nan

        penalty = 0.0
        notes = []
        if args.has_neg_column and bool(row.get("HasNegFlag", False)):
            penalty += 0.6
            notes.append("negflag")

        v_med, _ = robust_stats(arr)
        if np.isfinite(price) and np.isfinite(v_med) and v_med > 0 and price < 0.45 * v_med:
            penalty += 0.5
            notes.append("very_low_vs_variant_median")

        score = clamp(z - penalty, -10, 10) if np.isfinite(z) else np.nan
        deal_scores.append(score)
        deal_notes.append("|".join(notes))

    df_new["ProductId"] = assigned_pids
    df_new["VariantId"] = assigned_vids
    df_new["VariantText"] = variant_texts
    df_new["DealScore"] = deal_scores
    df_new["DealNotes"] = deal_notes

    # write/append output
    out_path = args.out
    if args.append and os.path.exists(out_path):
        df_prev = pd.read_csv(out_path)
        df_out = pd.concat([df_prev, df_new], ignore_index=True)
        df_out.to_csv(out_path, index=False)
    else:
        df_new.to_csv(out_path, index=False)

    print("✅ Incremental processed:", len(df_new))
    print(f"- Output: {out_path}")
    print(f"- Unassigned products: {(df_new['ProductId'] == -1).sum()}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite index created by batch script")
    ap.add_argument("--out", required=True, help="CSV to write assigned new items to (can append)")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--new_items", help="CSV containing only new items from this run (recommended)")
    g.add_argument("--growing_csv", help="Growing CSV; script will process only appended rows using DB pointer")

    ap.add_argument("--append", action="store_true", help="Append to --out if exists")
    ap.add_argument("--price_buffer_size", type=int, default=200)
    ap.add_argument("--has_neg_column", action="store_true", help="If your input new_items already has HasNegFlag column")
    return ap.parse_args()

if __name__ == "__main__":
    run(parse_args())
