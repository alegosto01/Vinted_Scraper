#!/usr/bin/env python3
"""
vinted_pipeline_batch.py

Batch mode:
- product clustering
- variant subclustering (autotune optional)
- deal scoring + trust report + outputs
- writes Product/Variant index into SQLite so incremental assignment is stable

Install:
  python -m pip install numpy pandas scikit-learn sentence-transformers matplotlib

Run:
  python vinted_pipeline_batch.py --input sold_df.csv --out_dir ./out --db ./out/index.sqlite --autotune_variants --make_plots
"""

import argparse
import os
import re
import math
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

import vinted_index_score as score

# ---------- text utils ----------
ALIAS_MAP = {
    r"\baf\s?1\b": "air force 1",
    r"\baf1\b": "air force 1",
    r"\blv\b": "louis vuitton",
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MODEL_CODE_RE = re.compile(r"\b[a-z]{1,4}\d{2,6}\b|\b\d{2,4}\b", re.IGNORECASE)

DEFAULT_NEGATIVE_KEYWORDS = [
    "replica","fake","falso","tarocco","ispirato","inspired","like","stile",
    "lotto","bundle","stock","solo scatola","scatola","dustbag","ricambio","solo",
    "rovinato","strappato","macchia","difetto","da riparare","rott",
    "kids","bambino","bimbi","ragazzo","ragazza",
]

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

def extract_model_codes(title_norm: str, max_codes=5):
    codes = MODEL_CODE_RE.findall(title_norm)
    seen = set()
    out = []
    for c in codes:
        c = c.lower()
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= max_codes:
            break
    return out

def make_block_key(row):
    brand = normalize_text(row.get("Brand", "")) or "nobrand"
    title = row.get("Title_norm", "")
    codes = extract_model_codes(title)
    if codes:
        return f"{brand}__codes__{'-'.join(codes)}"
    toks = tokenize(title)[:2]
    if toks:
        return f"{brand}__toks__{'-'.join(toks)}"
    return f"{brand}__notoks"

def negative_flags(title_norm: str, neg_keywords):
    return [kw for kw in neg_keywords if kw and kw in title_norm]

# ---------- embeddings ----------
def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    def embed(texts):
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    return embed, f"sentence-transformers:{model_name}"

# ---------- clustering ----------
def cluster_agglomerative_cosine(X, distance_threshold: float):
    from sklearn.cluster import AgglomerativeClustering
    n = X.shape[0]
    if n == 1:
        return np.array([0], dtype=int)
    cl = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return cl.fit_predict(X)

def choose_canonical_name(titles_norm):
    all_tokens = []
    for t in titles_norm:
        all_tokens.extend(tokenize(t))
    if not all_tokens:
        return "(no-title)"
    counts = Counter(all_tokens)
    top = [w for w, _ in counts.most_common(10)]
    return " ".join(top[:10])

def representative_index(X, idxs):
    V = X[idxs]
    centroid = V.mean(axis=0, keepdims=True)
    sims = (V @ centroid.T).reshape(-1)
    return idxs[int(np.argmax(sims))]

# ---------- variant subclustering ----------
def build_core_token_counts(titles_norm):
    # counts presence per listing (for stability)
    counts = {}
    for t in titles_norm:
        for tok in set(tokenize(t)):
            counts[tok] = counts.get(tok, 0) + 1
    return counts

def core_tokens_from_counts(counts, n_listings: int, min_frac: float):
    return {t for t, c in counts.items() if (c / max(1, n_listings)) >= min_frac}

def make_variant_text(title_norm: str, core: set):
    toks = tokenize(title_norm)
    return " ".join([t for t in toks if t not in core])

def robust_stats(x):
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(med), float(mad)

def autotune_variant_params(df_prod: pd.DataFrame):
    titles = df_prod["Title_norm"].fillna("").tolist()
    n = len(titles)

    p = pd.to_numeric(df_prod.get("Price"), errors="coerce").to_numpy(dtype=np.float32)
    p = p[np.isfinite(p) & (p > 0)]
    med_p, mad_p = robust_stats(p)
    spread_ratio = (mad_p / med_p) if (np.isfinite(med_p) and med_p > 0 and np.isfinite(mad_p)) else np.nan

    # start defaults
    core_frac = 0.70
    variant_threshold = 0.33
    price_weight = 0.35

    # empty-rate check
    counts = build_core_token_counts(titles)
    core = core_tokens_from_counts(counts, n, core_frac)
    vt = [make_variant_text(t, core) for t in titles]
    empty_rate = float(np.mean([1.0 if not s.strip() else 0.0 for s in vt]))

    if empty_rate > 0.55: core_frac = 0.60
    elif empty_rate > 0.40: core_frac = 0.65
    elif empty_rate < 0.15: core_frac = 0.78

    if np.isfinite(spread_ratio):
        if spread_ratio >= 0.22:
            price_weight = 0.55; variant_threshold = 0.30
        elif spread_ratio >= 0.14:
            price_weight = 0.45; variant_threshold = 0.31
        elif spread_ratio <= 0.06:
            price_weight = 0.25; variant_threshold = 0.36

    return core_frac, variant_threshold, price_weight

def variant_cluster(df_prod, embed, core_frac, vthr, pwt):
    titles = df_prod["Title_norm"].fillna("").tolist()
    counts = build_core_token_counts(titles)
    core = core_tokens_from_counts(counts, len(titles), core_frac)
    vtexts = [make_variant_text(t, core) for t in titles]
    V = embed(vtexts)

    p = pd.to_numeric(df_prod.get("Price"), errors="coerce").to_numpy(dtype=np.float32)
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan).astype(np.float32)
    lp = np.log(p)
    lp = (lp - np.nanmean(lp)) / (np.nanstd(lp) + 1e-6)
    lp = np.nan_to_num(lp, nan=0.0).astype(np.float32)

    Xv = np.hstack([V, (pwt * lp).reshape(-1, 1)]).astype(np.float32)
    labs = cluster_agglomerative_cosine(Xv, distance_threshold=vthr) if len(df_prod) > 1 else np.array([0], dtype=int)
    return vtexts, labs, core, counts, Xv

def deal_score_variant(prices: np.ndarray, price: float):
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 3 or not np.isfinite(price) or price <= 0:
        return np.nan
    med = np.median(prices)
    mad = np.median(np.abs(prices - med))
    scale = 1.4826 * mad + 1e-6
    return float((med - price) / scale)

# ---------- plots ----------
def plot_variant_prices(df_prod, product_id, plots_dir):
    import matplotlib.pyplot as plt
    dfp = df_prod.copy()
    dfp["_PriceNum"] = pd.to_numeric(dfp.get("Price"), errors="coerce")
    med_by = dfp.groupby("VariantId")["_PriceNum"].median().sort_values()
    vars_order = med_by.index.tolist()

    data, labels = [], []
    for vid in vars_order:
        vals = dfp.loc[dfp["VariantId"] == vid, "_PriceNum"].dropna().values
        if len(vals):
            data.append(vals); labels.append(str(vid))

    if not data:
        return
    plt.figure(figsize=(max(8, len(data)*0.8), 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"Product {product_id} - Variant price distributions")
    plt.xlabel("VariantId"); plt.ylabel("Price")
    out = os.path.join(plots_dir, f"product_{product_id}_variant_prices.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

def plot_variant_pca(Xv, labs, product_id, plots_dir):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    if Xv.shape[0] < 3:
        return
    Y = PCA(n_components=2, random_state=0).fit_transform(Xv)
    plt.figure(figsize=(6.8, 5.2))
    for lab in np.unique(labs):
        idx = (labs == lab)
        plt.scatter(Y[idx,0], Y[idx,1], label=str(lab), s=18)
    plt.title(f"Product {product_id} - Variant PCA")
    plt.legend(fontsize=8, loc="best")
    out = os.path.join(plots_dir, f"product_{product_id}_variant_pca.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

# ---------- main ----------
def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    if args.make_plots:
        os.makedirs(plots_dir, exist_ok=True)

    con = score.connect(args.db)

    df = pd.read_csv(args.input)
    df["Title_norm"] = df["Title"].apply(normalize_text)
    df["Brand_norm"] = df["Brand"].apply(normalize_text) if "Brand" in df.columns else ""
    df["NegFlags"] = df["Title_norm"].apply(lambda t: "|".join(negative_flags(t, DEFAULT_NEGATIVE_KEYWORDS)))
    df["HasNegFlag"] = df["NegFlags"].astype(str).apply(lambda s: s not in ("", "nan"))

    df["BlockKey"] = df.apply(make_block_key, axis=1)
    size_col = df["Size"].astype(str) if "Size" in df.columns else ""
    df["EmbedText"] = (df["Title_norm"].fillna("") + " | " + df["Brand_norm"].fillna("") + " | " + size_col.fillna("")).astype(str)

    embed, embedder_name = get_embedder(args.model)
    X = embed(df["EmbedText"].tolist())

    # Product clustering per block
    product_id = np.full(len(df), -1, dtype=int)
    product_meta = {}
    next_pid = 0

    for block_key, idxs in df.groupby("BlockKey").indices.items():
        idxs = list(idxs)
        if len(idxs) == 1:
            pid = next_pid; next_pid += 1
            product_id[idxs[0]] = pid
            product_meta[pid] = (block_key, choose_canonical_name([df.loc[idxs[0], "Title_norm"]]), int(idxs[0]), 1)
            continue

        labs = cluster_agglomerative_cosine(X[idxs], distance_threshold=args.product_threshold)
        for lab in np.unique(labs):
            members = [idxs[i] for i in range(len(idxs)) if labs[i] == lab]
            pid = next_pid; next_pid += 1
            for m in members: product_id[m] = pid
            rep_i = representative_index(X, members)
            cname = choose_canonical_name(df.loc[members, "Title_norm"].tolist())
            product_meta[pid] = (block_key, cname, int(rep_i), len(members))

    df["ProductId"] = product_id

    # Variant clustering + export index
    next_vid = 0
    df_parts = []
    variant_rows = []
    trust_rows = []
    deals_rows = []

    for pid, g in df.groupby("ProductId"):
        g = g.copy()
        block_key, cname, rep_i, nprod = product_meta[int(pid)]

        # product centroid & core token counts
        idxs = g.index.to_list()
        centroid = X[idxs].mean(axis=0)
        core_counts = build_core_token_counts(g["Title_norm"].tolist())

        # decide core_frac/vthr/pwt
        if args.autotune_variants and len(g) >= args.min_product_size_for_variants:
            core_frac, vthr, pwt = autotune_variant_params(g)
        else:
            core_frac, vthr, pwt = args.core_frac, args.variant_threshold, args.variant_price_weight

        # save product to DB
        score.upsert_product(
            con,
            product_id=int(pid),
            centroid=centroid,
            n=int(len(g)),
            product_threshold=float(args.product_threshold),
            canonical_name=cname,
            block_key_hint=block_key,
            core_token_counts=core_counts,
            core_token_min_frac=float(core_frac),
        )

        # if too small => single variant
        if len(g) < args.min_product_size_for_variants:
            g["VariantText"] = ""
            g["VariantId_local"] = 0
            g["VariantId"] = next_vid
            # variant centroid = embed("") basically: use product centroid slice as proxy (ok)
            vcentroid = np.zeros(X.shape[1] + 1, dtype=np.float32)
            score.upsert_variant(con, next_vid, int(pid), vcentroid, int(len(g)), float(vthr), float(pwt), float(core_frac), [], "")
            next_vid += 1
            df_parts.append(g)
            continue

        vtexts, vlabs, core_tokens, _, Xv = variant_cluster(g, embed, core_frac, vthr, pwt)
        g["VariantText"] = vtexts
        g["VariantId_local"] = vlabs

        # map local to global VariantId
        local_to_global = {}
        for lab in sorted(np.unique(vlabs)):
            local_to_global[int(lab)] = next_vid
            next_vid += 1
        g["VariantId"] = g["VariantId_local"].map(local_to_global).astype(int)

        # trust silhouette (optional)
        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(Xv, vlabs, metric="cosine")) if len(np.unique(vlabs)) >= 2 else np.nan
        except Exception:
            sil = np.nan
        trust_rows.append({"ProductId": int(pid), "n": int(len(g)), "VariantsFound": int(len(np.unique(vlabs))), "VariantSilhouette": sil})

        # score variants in DB
        for lab in sorted(np.unique(vlabs)):
            members = g.index[g["VariantId_local"] == lab].to_list()
            vid = local_to_global[int(lab)]
            vcentroid = Xv[[g.index.get_loc(i) for i in members]].mean(axis=0)
            prices = pd.to_numeric(g.loc[members, "Price"], errors="coerce").dropna().tolist()
            # keep last N prices in buffer
            buf = prices[-args.price_buffer_size:]
            # top tokens of variant text (for inspect)
            vtoks = []
            for s in g.loc[members, "VariantText"].fillna("").tolist():
                vtoks.extend(tokenize(s))
            top_v = " ".join([w for w, _ in Counter(vtoks).most_common(8)])

            score.upsert_variant(con, int(vid), int(pid), vcentroid, int(len(members)), float(vthr), float(pwt), float(core_frac), buf, top_v)

            # variant summary row
            pnum = pd.to_numeric(g.loc[members, "Price"], errors="coerce").to_numpy()
            variant_rows.append({
                "ProductId": int(pid), "VariantId": int(vid), "VariantCount": int(len(members)),
                "VariantTextTop": top_v,
                "PriceMedian": float(np.nanmedian(pnum)) if np.isfinite(pnum).any() else np.nan,
                "PriceMin": float(np.nanmin(pnum)) if np.isfinite(pnum).any() else np.nan,
                "PriceMax": float(np.nanmax(pnum)) if np.isfinite(pnum).any() else np.nan,
                "CoreFrac": float(core_frac), "VariantThreshold": float(vthr), "PriceWeight": float(pwt),
            })

        # plots
        if args.make_plots:
            plot_variant_prices(g, int(pid), plots_dir)
            plot_variant_pca(Xv, vlabs, int(pid), plots_dir)

        df_parts.append(g)

    df2 = pd.concat(df_parts, ignore_index=True)

    # Deal scoring per variant
    df2["_PriceNum"] = pd.to_numeric(df2.get("Price"), errors="coerce")
    prices_by_variant = {int(vid): grp["_PriceNum"].to_numpy(dtype=np.float32) for vid, grp in df2.groupby("VariantId")}

    def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

    scores, notes = [], []
    for _, row in df2.iterrows():
        vid = int(row["VariantId"])
        price = float(row["_PriceNum"]) if np.isfinite(row["_PriceNum"]) else np.nan
        z = deal_score_variant(prices_by_variant.get(vid, np.array([])), price)
        penalty = 0.0
        nn = []
        if bool(row.get("HasNegFlag", False)):
            penalty += 0.6; nn.append("negflag")
        v_med, _ = robust_stats(prices_by_variant.get(vid, np.array([])))
        if np.isfinite(price) and np.isfinite(v_med) and v_med > 0 and price < 0.45 * v_med:
            penalty += 0.5; nn.append("very_low_vs_variant_median")
        sc = clamp(z - penalty, -10, 10) if np.isfinite(z) else np.nan
        scores.append(sc); notes.append("|".join(nn))

    df2["DealScore"] = scores
    df2["DealNotes"] = notes

    # outputs
    items_out = os.path.join(args.out_dir, "items_with_product_and_variant.csv")
    var_out = os.path.join(args.out_dir, "variant_summary.csv")
    trust_out = os.path.join(args.out_dir, "trust_report.csv")
    deals_out = os.path.join(args.out_dir, "deals_ranked.csv")

    df2.to_csv(items_out, index=False)
    pd.DataFrame(variant_rows).to_csv(var_out, index=False)
    pd.DataFrame(trust_rows).to_csv(trust_out, index=False)

    deals_ranked = df2.copy()
    # require variant size >= min_variant_size_for_deals
    var_sizes = deals_ranked.groupby("VariantId").size().to_dict()
    deals_ranked["VariantClusterSize"] = deals_ranked["VariantId"].map(var_sizes)
    deals_ranked = deals_ranked[deals_ranked["VariantClusterSize"] >= args.min_variant_size_for_deals]
    deals_ranked = deals_ranked.sort_values("DealScore", ascending=False)
    deals_ranked.to_csv(deals_out, index=False)

    score.set_meta(con, "embedder", embedder_name)
    score.set_meta(con, "model_name", args.model)

    print("✅ Batch complete")
    print(f"- Index DB: {args.db}")
    print(f"- Items:    {items_out}")
    print(f"- Deals:    {deals_out}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="./out")
    ap.add_argument("--db", default="./out/index.sqlite")

    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--product_threshold", type=float, default=0.24)

    ap.add_argument("--autotune_variants", action="store_true")
    ap.add_argument("--variant_threshold", type=float, default=0.33)
    ap.add_argument("--core_frac", type=float, default=0.70)
    ap.add_argument("--variant_price_weight", type=float, default=0.35)

    ap.add_argument("--min_product_size_for_variants", type=int, default=4)
    ap.add_argument("--min_variant_size_for_deals", type=int, default=4)

    ap.add_argument("--price_buffer_size", type=int, default=200)

    ap.add_argument("--make_plots", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    run(parse_args())
