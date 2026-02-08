#!/usr/bin/env python3
"""
vinted_cluster_autotune_deals_viz.py

Adds 3 things to the previous pipeline:
A) Auto-tuning variant subclustering parameters PER PRODUCT (core_frac, variant_threshold, price_weight)
B) Deal scoring PER VARIANT (robust, price-based, with penalties)
C) Visualization outputs (PNG) so you can trust clusters before automating buys

Outputs (in --out_dir):
- items_with_product_and_variant.csv
- product_summary.csv
- variant_summary.csv
- deals_ranked.csv
- trust_report.csv
- plots/
    - product_<ProductId>_variant_prices.png
    - product_<ProductId>_variant_pca.png

Install:
  python -m pip install numpy pandas scikit-learn sentence-transformers matplotlib

Run:
  python vinted_cluster_autotune_deals_viz.py --input /mnt/data/sold_df.csv --out_dir ./out
"""

import argparse
import os
import re
import math
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# -----------------------------
# Normalization / tokenization
# -----------------------------

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

# -----------------------------
# Blocking (brand + model codes)
# -----------------------------

MODEL_CODE_RE = re.compile(r"\b[a-z]{1,4}\d{2,6}\b|\b\d{2,4}\b", re.IGNORECASE)

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

# -----------------------------
# Soft negative tags (optional)
# -----------------------------

DEFAULT_NEGATIVE_KEYWORDS = [
    "replica","fake","falso","tarocco","ispirato","inspired","like","stile",
    "lotto","bundle","stock","solo scatola","scatola","dustbag","ricambio","solo",
    "rovinato","strappato","macchia","difetto","da riparare","rott",
    "kids","bambino","bimbi","ragazzo","ragazza",
]

def negative_flags(title_norm: str, neg_keywords):
    flags = []
    for kw in neg_keywords:
        if kw and kw in title_norm:
            flags.append(kw)
    return flags

# -----------------------------
# Embeddings
# -----------------------------

def get_embedder(model_name: str):
    """
    Returns (embed_func, embedder_name)
    embed_func(texts)-> np.ndarray [n,d] normalized OR sparse tfidf matrix fallback
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)

        def embed(texts):
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)

        return embed, f"sentence-transformers:{model_name}"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))

        def embed(texts):
            X = vectorizer.fit_transform(texts)
            X = normalize(X)
            return X

        return embed, "tfidf-fallback"

# -----------------------------
# Clustering helpers
# -----------------------------

def cluster_agglomerative_cosine(vectors, distance_threshold: float):
    from sklearn.cluster import AgglomerativeClustering

    n = vectors.shape[0]
    if n == 1:
        return np.array([0], dtype=int)

    is_sparse = hasattr(vectors, "toarray")
    if is_sparse:
        if n <= 2000 and vectors.shape[1] <= 8000:
            X = vectors.toarray().astype(np.float32)
        else:
            raise RuntimeError(
                "TF-IDF fallback produced a large sparse matrix. Install sentence-transformers for scale."
            )
    else:
        X = vectors

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

def representative_index(vectors, idxs):
    is_sparse = hasattr(vectors, "toarray")
    if is_sparse:
        X = vectors[idxs].toarray().astype(np.float32)
    else:
        X = vectors[idxs]
    centroid = X.mean(axis=0, keepdims=True)
    sims = (X @ centroid.T).reshape(-1)
    return idxs[int(np.argmax(sims))]

# -----------------------------
# Variant subclustering (automatic)
# -----------------------------

def build_core_tokens(titles_norm, min_frac=0.7):
    token_lists = [tokenize(t) for t in titles_norm]
    n = len(token_lists)
    if n == 0:
        return set()
    counts = {}
    for toks in token_lists:
        for t in set(toks):
            counts[t] = counts.get(t, 0) + 1
    return {t for t, c in counts.items() if (c / n) >= min_frac}

def make_variant_text(title_norm: str, core_tokens: set):
    toks = tokenize(title_norm)
    var = [t for t in toks if t not in core_tokens]
    return " ".join(var)

def robust_stats(x: np.ndarray):
    """median + MAD (robust scale)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(med), float(mad)

def autotune_variant_params(df_prod: pd.DataFrame):
    """
    Auto-tune per ProductId using cheap heuristics.
    Returns (core_frac, variant_threshold, price_weight, notes_dict).

    Intuition:
    - if price spread is high => weight price more and split a bit more (lower threshold)
    - if variant_text is often empty => core_frac too high -> lower core_frac
    - if titles are very uniform => rely more on price (higher price_weight)
    """
    titles = df_prod["Title_norm"].fillna("").tolist()
    n = len(titles)

    # price spread
    p = pd.to_numeric(df_prod.get("Price", pd.Series([np.nan]*n)), errors="coerce").to_numpy(dtype=np.float32)
    p = p[np.isfinite(p) & (p > 0)]
    med_p, mad_p = robust_stats(p)
    spread_ratio = np.nan
    if np.isfinite(med_p) and med_p > 0 and np.isfinite(mad_p):
        spread_ratio = mad_p / med_p  # robust coefficient of variation proxy

    # title uniformity proxy: average Jaccard similarity to most-common token set
    token_sets = [set(tokenize(t)) for t in titles]
    all_tokens = []
    for s in token_sets:
        all_tokens.extend(list(s))
    counts = Counter(all_tokens)
    common = {t for t, c in counts.items() if c / max(1, n) >= 0.7}

    def jacc(a, b):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    jaccs = [jacc(s, common) for s in token_sets]
    uniformity = float(np.mean(jaccs)) if jaccs else 0.0  # higher => more similar titles

    # start defaults
    core_frac = 0.70
    variant_threshold = 0.33
    price_weight = 0.35

    # adjust core_frac based on emptiness of variant text
    # We'll simulate with current core_frac, then adjust.
    core = build_core_tokens(titles, min_frac=core_frac)
    vt = [make_variant_text(t, core) for t in titles]
    empty_rate = float(np.mean([1.0 if (not s.strip()) else 0.0 for s in vt])) if n else 1.0

    # If too many empties => core too aggressive
    if empty_rate > 0.55:
        core_frac = 0.60
    elif empty_rate > 0.40:
        core_frac = 0.65
    elif empty_rate < 0.15:
        core_frac = 0.78  # keep more core removal to focus on extras

    # price influence
    if np.isfinite(spread_ratio):
        if spread_ratio >= 0.22:  # big spread => bundles likely
            price_weight = 0.55
            variant_threshold = 0.30
        elif spread_ratio >= 0.14:
            price_weight = 0.45
            variant_threshold = 0.31
        elif spread_ratio <= 0.06:  # little spread => variants less likely
            price_weight = 0.25
            variant_threshold = 0.36

    # if titles are extremely uniform, rely more on price
    if uniformity > 0.72:
        price_weight = min(0.65, price_weight + 0.10)
        variant_threshold = max(0.28, variant_threshold - 0.01)

    notes = {
        "n": n,
        "price_median": med_p,
        "price_mad": mad_p,
        "spread_ratio_mad_over_median": spread_ratio,
        "title_uniformity": uniformity,
        "variant_empty_rate_initial": empty_rate,
        "autotuned_core_frac": core_frac,
        "autotuned_variant_threshold": variant_threshold,
        "autotuned_price_weight": price_weight,
    }
    return core_frac, variant_threshold, price_weight, notes

def variant_subcluster_for_product(df_prod, embed_func, variant_threshold=0.33, core_frac=0.7, price_weight=0.35):
    titles = df_prod["Title_norm"].fillna("").tolist()
    core = build_core_tokens(titles, min_frac=core_frac)
    variant_texts = [make_variant_text(t, core) for t in titles]

    V = embed_func(variant_texts)

    # add log(price) dim
    n = len(df_prod)
    p = pd.to_numeric(df_prod.get("Price", pd.Series([np.nan]*n)), errors="coerce").to_numpy(dtype=np.float32)
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan).astype(np.float32)
    lp = np.log(p)
    lp = (lp - np.nanmean(lp)) / (np.nanstd(lp) + 1e-6)
    lp = np.nan_to_num(lp, nan=0.0).astype(np.float32)

    is_sparse = hasattr(V, "toarray")
    if is_sparse:
        X = V.toarray().astype(np.float32)
    else:
        X = V

    X = np.hstack([X, (price_weight * lp).reshape(-1, 1)]).astype(np.float32)

    labels = cluster_agglomerative_cosine(X, distance_threshold=variant_threshold) if n > 1 else np.array([0], dtype=int)

    out = df_prod.copy()
    out["VariantText"] = variant_texts
    out["VariantId_local"] = labels
    out["CoreFracUsed"] = core_frac
    out["VariantThresholdUsed"] = variant_threshold
    out["VariantPriceWeightUsed"] = price_weight
    return out, X  # return X for viz/silhouette if needed

# -----------------------------
# Deal scoring per VariantId
# -----------------------------

def deal_score_for_variant(prices: np.ndarray, price: float):
    """
    Robust deal score:
      z = (median - price) / (1.4826*MAD + eps)
    Higher z => better deal (cheaper vs typical).
    """
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 3 or (not np.isfinite(price)) or price <= 0:
        return np.nan

    med = np.median(prices)
    mad = np.median(np.abs(prices - med))
    scale = 1.4826 * mad + 1e-6
    z = (med - price) / scale
    return float(z)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# -----------------------------
# Visualization / trust report
# -----------------------------

def ensure_plots_dir(out_dir):
    p = os.path.join(out_dir, "plots")
    os.makedirs(p, exist_ok=True)
    return p

def plot_variant_prices(df_prod: pd.DataFrame, product_id: int, plots_dir: str):
    import matplotlib.pyplot as plt

    # sort variants by median price
    price_series = pd.to_numeric(df_prod.get("Price", pd.Series([np.nan]*len(df_prod))), errors="coerce")
    df_plot = df_prod.copy()
    df_plot["_PriceNum"] = price_series

    med_by_var = df_plot.groupby("VariantId")["_PriceNum"].median().sort_values()
    ordered_vars = med_by_var.index.tolist()

    data = []
    labels = []
    for vid in ordered_vars:
        vals = df_plot.loc[df_plot["VariantId"] == vid, "_PriceNum"].dropna().values
        if len(vals) >= 1:
            data.append(vals)
            labels.append(str(vid))

    if not data:
        return

    plt.figure(figsize=(max(8, len(data)*0.8), 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"Product {product_id} - Variant price distributions")
    plt.xlabel("VariantId")
    plt.ylabel("Price")
    outpath = os.path.join(plots_dir, f"product_{product_id}_variant_prices.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_variant_pca(X_variant: np.ndarray, labels: np.ndarray, product_id: int, plots_dir: str):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if X_variant.shape[0] < 3:
        return

    # 2D PCA
    pca = PCA(n_components=2, random_state=0)
    Y = pca.fit_transform(X_variant)

    plt.figure(figsize=(6.8, 5.2))
    for lab in np.unique(labels):
        idx = (labels == lab)
        plt.scatter(Y[idx, 0], Y[idx, 1], label=str(lab), s=18)
    plt.title(f"Product {product_id} - Variant embedding PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="VariantLocal", fontsize=8, loc="best")
    outpath = os.path.join(plots_dir, f"product_{product_id}_variant_pca.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def silhouette_safe(X: np.ndarray, labels: np.ndarray):
    try:
        from sklearn.metrics import silhouette_score
        # needs at least 2 clusters, no singletons-only
        if len(np.unique(labels)) < 2:
            return np.nan
        # silhouette fails if a cluster has 1 sample sometimes; still usually ok
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return np.nan

# -----------------------------
# Main pipeline
# -----------------------------

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = ensure_plots_dir(args.out_dir)

    df = pd.read_csv(args.input)
    if "Title" not in df.columns:
        raise ValueError("Input CSV must contain a 'Title' column.")

    df["Title_norm"] = df["Title"].apply(normalize_text)
    if "Brand" in df.columns:
        df["Brand_norm"] = df["Brand"].apply(normalize_text)
    else:
        df["Brand"] = ""
        df["Brand_norm"] = ""

    # negative tags
    neg_keywords = DEFAULT_NEGATIVE_KEYWORDS[:]
    if args.neg.strip():
        neg_keywords.extend([normalize_text(x) for x in args.neg.split(",") if x.strip()])
    df["NegFlags"] = df["Title_norm"].apply(lambda t: "|".join(negative_flags(t, neg_keywords)))
    df["HasNegFlag"] = df["NegFlags"].astype(str).apply(lambda s: (s != "") and (s != "nan"))

    # blocking
    df["BlockKey"] = df.apply(make_block_key, axis=1)

    # embeddings for product clustering
    size_col = df["Size"].astype(str) if "Size" in df.columns else ""
    df["EmbedText"] = (df["Title_norm"].fillna("") + " | " + df["Brand_norm"].fillna("") + " | " + size_col.fillna("")).astype(str)

    embed, embedder_name = get_embedder(args.model)
    vectors = embed(df["EmbedText"].tolist())

    # product clustering per block
    product_id = np.full(len(df), -1, dtype=int)
    product_meta = {}
    next_pid = 0

    for block_key, idxs in df.groupby("BlockKey").indices.items():
        idxs = list(idxs)
        if len(idxs) == 1:
            pid = next_pid; next_pid += 1
            product_id[idxs[0]] = pid
            product_meta[pid] = {
                "BlockKey": block_key,
                "CanonicalName": choose_canonical_name([df.loc[idxs[0], "Title_norm"]]),
                "RepresentativeIndex": int(idxs[0]),
            }
            continue

        labs = cluster_agglomerative_cosine(vectors[idxs], distance_threshold=args.product_threshold)
        for lab in np.unique(labs):
            members = [idxs[i] for i in range(len(idxs)) if labs[i] == lab]
            pid = next_pid; next_pid += 1
            for m in members:
                product_id[m] = pid
            rep_i = representative_index(vectors, members)
            product_meta[pid] = {
                "BlockKey": block_key,
                "CanonicalName": choose_canonical_name(df.loc[members, "Title_norm"].tolist()),
                "RepresentativeIndex": int(rep_i),
            }

    df["ProductId"] = product_id

    # product summary
    product_rows = []
    for pid, g in df.groupby("ProductId"):
        prices = pd.to_numeric(g.get("Price", pd.Series([np.nan]*len(g))), errors="coerce")
        vv = g.get("Brand", pd.Series([], dtype=str)).dropna().astype(str)
        brand_mode = vv.value_counts().index[0] if len(vv) else ""

        meta = product_meta.get(pid, {})
        rep_idx = meta.get("RepresentativeIndex", int(g.index[0]))
        rep_row = df.loc[rep_idx]

        product_rows.append({
            "ProductId": int(pid),
            "CanonicalName": meta.get("CanonicalName", choose_canonical_name(g["Title_norm"].tolist())),
            "BrandMode": brand_mode,
            "Count": int(len(g)),
            "PriceMedian": float(np.nanmedian(prices)) if prices.notna().any() else np.nan,
            "PriceMin": float(np.nanmin(prices)) if prices.notna().any() else np.nan,
            "PriceMax": float(np.nanmax(prices)) if prices.notna().any() else np.nan,
            "NegFlagRate": float(g["HasNegFlag"].mean()),
            "RepresentativeTitle": rep_row.get("Title", ""),
            "RepresentativeLink": rep_row.get("Link", ""),
            "BlockKey": meta.get("BlockKey", ""),
            "Embedder": embedder_name,
            "ProductThreshold": float(args.product_threshold),
        })

    product_summary = pd.DataFrame(product_rows).sort_values(["Count", "PriceMedian"], ascending=[False, True])

    # -----------------------------------------
    # Variant subclustering + autotuning + viz
    # -----------------------------------------
    pieces = []
    trust_rows = []
    variant_rows = []
    next_global_variant = 0

    for pid, g in df.groupby("ProductId"):
        g = g.copy()

        # If small, skip variants
        if len(g) < args.min_product_size_for_variants:
            g["VariantText"] = ""
            g["VariantId_local"] = 0
            g["VariantId"] = next_global_variant
            next_global_variant += 1

            prices = pd.to_numeric(g.get("Price", pd.Series([np.nan]*len(g))), errors="coerce")
            variant_rows.append({
                "ProductId": int(pid),
                "VariantId": int(g["VariantId"].iloc[0]),
                "VariantLocal": 0,
                "VariantCount": int(len(g)),
                "VariantTextTop": "",
                "PriceMedian": float(np.nanmedian(prices)) if prices.notna().any() else np.nan,
                "PriceMin": float(np.nanmin(prices)) if prices.notna().any() else np.nan,
                "PriceMax": float(np.nanmax(prices)) if prices.notna().any() else np.nan,
                "Autotuned": False,
                "VariantThreshold": np.nan,
                "CoreFrac": np.nan,
                "PriceWeight": np.nan,
            })

            trust_rows.append({
                "ProductId": int(pid),
                "Autotuned": False,
                "n": int(len(g)),
                "VariantsFound": 1,
                "VariantSilhouette": np.nan,
                "Notes": "skipped_small_product_cluster",
            })
            pieces.append(g)
            continue

        # autotune (or fixed)
        if args.autotune_variants:
            core_frac, vthr, pwt, notes = autotune_variant_params(g)
            autotuned = True
        else:
            core_frac, vthr, pwt = args.core_frac, args.variant_threshold, args.variant_price_weight
            notes = {"autotune_disabled": True}
            autotuned = False

        g2, X_variant = variant_subcluster_for_product(
            g,
            embed_func=embed,
            variant_threshold=vthr,
            core_frac=core_frac,
            price_weight=pwt,
        )

        # map local -> global VariantId
        local_to_global = {}
        for lab in sorted(g2["VariantId_local"].unique()):
            local_to_global[int(lab)] = next_global_variant
            next_global_variant += 1

        g2["VariantId"] = g2["VariantId_local"].map(local_to_global).astype(int)

        # trust: silhouette on combined (variant embedding + price dim)
        sil = silhouette_safe(X_variant, g2["VariantId_local"].to_numpy())
        trust_rows.append({
            "ProductId": int(pid),
            "Autotuned": autotuned,
            "n": int(len(g2)),
            "VariantsFound": int(g2["VariantId_local"].nunique()),
            "VariantSilhouette": sil,
            "Notes": str(notes),
        })

        # viz for this product (optional)
        if args.make_plots:
            plot_variant_prices(g2, int(pid), plots_dir)
            plot_variant_pca(X_variant, g2["VariantId_local"].to_numpy(), int(pid), plots_dir)

        # variant summaries for this product
        for lab, gg in g2.groupby("VariantId_local"):
            prices = pd.to_numeric(gg.get("Price", pd.Series([np.nan]*len(gg))), errors="coerce")
            vtoks = []
            for s in gg["VariantText"].fillna("").tolist():
                vtoks.extend(tokenize(s))
            top_v = " ".join([w for w, _ in Counter(vtoks).most_common(8)])

            variant_rows.append({
                "ProductId": int(pid),
                "VariantId": int(local_to_global[int(lab)]),
                "VariantLocal": int(lab),
                "VariantCount": int(len(gg)),
                "VariantTextTop": top_v,
                "PriceMedian": float(np.nanmedian(prices)) if prices.notna().any() else np.nan,
                "PriceMin": float(np.nanmin(prices)) if prices.notna().any() else np.nan,
                "PriceMax": float(np.nanmax(prices)) if prices.notna().any() else np.nan,
                "Autotuned": autotuned,
                "VariantThreshold": float(vthr),
                "CoreFrac": float(core_frac),
                "PriceWeight": float(pwt),
            })

        pieces.append(g2)

    df2 = pd.concat(pieces, ignore_index=True)
    variant_summary = pd.DataFrame(variant_rows).sort_values(["VariantCount", "PriceMedian"], ascending=[False, True])
    trust_report = pd.DataFrame(trust_rows).sort_values(["VariantsFound", "VariantSilhouette"], ascending=[False, False])

    # Attach names/sizes
    prod_size_map = product_summary.set_index("ProductId")["Count"].to_dict()
    prod_name_map = product_summary.set_index("ProductId")["CanonicalName"].to_dict()
    df2["ProductClusterSize"] = df2["ProductId"].map(prod_size_map)
    df2["ProductCanonicalName"] = df2["ProductId"].map(prod_name_map)

    var_size_map = variant_summary.set_index("VariantId")["VariantCount"].to_dict()
    df2["VariantClusterSize"] = df2["VariantId"].map(var_size_map)

    # -----------------------------------------
    # Deal scoring per VariantId (robust z-score)
    # -----------------------------------------
    df2["_PriceNum"] = pd.to_numeric(df2.get("Price", pd.Series([np.nan]*len(df2))), errors="coerce")
    # precompute variant price arrays
    prices_by_variant = {}
    for vid, g in df2.groupby("VariantId"):
        prices_by_variant[int(vid)] = g["_PriceNum"].to_numpy(dtype=np.float32)

    deal_scores = []
    deal_notes = []
    for i, row in df2.iterrows():
        vid = int(row["VariantId"])
        price = float(row["_PriceNum"]) if np.isfinite(row["_PriceNum"]) else np.nan
        z = deal_score_for_variant(prices_by_variant.get(vid, np.array([])), price)

        # penalties / guards
        penalty = 0.0
        note_parts = []

        if bool(row.get("HasNegFlag", False)):
            penalty += 0.6
            note_parts.append("negflag")

        # if it's absurdly low vs median => likely incomplete/scam: still score, but flag it
        v_prices = prices_by_variant.get(vid, np.array([]))
        v_med, v_mad = robust_stats(v_prices[np.isfinite(v_prices)])
        if np.isfinite(price) and np.isfinite(v_med) and v_med > 0:
            if price < 0.45 * v_med:
                penalty += 0.5
                note_parts.append("very_low_vs_variant_median")

        # final score
        if np.isfinite(z):
            score = z - penalty
            score = clamp(score, -10.0, 10.0)
        else:
            score = np.nan

        deal_scores.append(score)
        deal_notes.append("|".join(note_parts))

    df2["DealScore"] = deal_scores
    df2["DealNotes"] = deal_notes

    # ranked deals view
    deals_ranked = df2.copy()
    # require some minimum variant size to avoid noise
    deals_ranked = deals_ranked[deals_ranked["VariantClusterSize"].fillna(0) >= args.min_variant_size_for_deals]
    deals_ranked = deals_ranked.sort_values(["DealScore"], ascending=False)

    # -----------------------------------------
    # Save everything
    # -----------------------------------------
    items_out = os.path.join(args.out_dir, "items_with_product_and_variant.csv")
    prod_out  = os.path.join(args.out_dir, "product_summary.csv")
    var_out   = os.path.join(args.out_dir, "variant_summary.csv")
    deals_out = os.path.join(args.out_dir, "deals_ranked.csv")
    trust_out = os.path.join(args.out_dir, "trust_report.csv")

    df2.to_csv(items_out, index=False)
    product_summary.to_csv(prod_out, index=False)
    variant_summary.to_csv(var_out, index=False)
    deals_ranked.to_csv(deals_out, index=False)
    trust_report.to_csv(trust_out, index=False)

    print("✅ Done.")
    print(f"- Items:        {items_out}")
    print(f"- Products:     {prod_out}   (n={df2['ProductId'].nunique()})")
    print(f"- Variants:     {var_out}   (n={df2['VariantId'].nunique()})")
    print(f"- Deals ranked: {deals_out}")
    print(f"- Trust report: {trust_out}")
    if args.make_plots:
        print(f"- Plots dir:    {plots_dir}")
    print(f"- Embedder:     {embedder_name}")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV (e.g., sold_df.csv)")
    ap.add_argument("--out_dir", default="./out", help="Output directory")

    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="Embedding model (multilingual default).")

    ap.add_argument("--product_threshold", type=float, default=0.24,
                    help="Cosine distance threshold for PRODUCT clustering. Typical: 0.20-0.30")

    # Variant parameters (used only when autotune is disabled)
    ap.add_argument("--variant_threshold", type=float, default=0.33,
                    help="Cosine distance threshold for VARIANT subclustering. Typical: 0.30-0.45")
    ap.add_argument("--core_frac", type=float, default=0.70,
                    help="Core token fraction inside a ProductId. Typical: 0.60-0.80")
    ap.add_argument("--variant_price_weight", type=float, default=0.35,
                    help="How much price influences variant clustering. Typical: 0.20-0.60")

    # Auto-tuning + plot toggles
    ap.add_argument("--autotune_variants", action="store_true",
                    help="Enable per-product auto-tuning for variant parameters.")
    ap.add_argument("--make_plots", action="store_true",
                    help="Save plots (PNG) for variant price distributions and PCA per product.")

    ap.add_argument("--min_product_size_for_variants", type=int, default=4,
                    help="Skip variant subclustering if product cluster is smaller than this.")
    ap.add_argument("--min_variant_size_for_deals", type=int, default=4,
                    help="Only rank deals for variants with at least this many listings.")
    ap.add_argument("--neg", default="",
                    help="Comma-separated extra negative keywords (tags only).")

    return ap.parse_args()

if __name__ == "__main__":
    run(parse_args())
