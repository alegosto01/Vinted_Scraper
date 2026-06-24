#!/usr/bin/env python3
"""
Build the batch deal-analysis dataset from scraped listings.

This script clusters listings into products and variants, computes deal-side
metrics like DealScore/ExpectedProfit/ResaleSafetyScore, and writes
outputs such as deals_ranked.csv for downstream filtering and evaluation.
"""

import argparse
import os
import re
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiments.old.clustering_approach.vinted_index_score as score
from analysis_pipeline.scoring.visual_rerank import image_from_source, normalize_image_sources

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

GENERIC_TITLE_TOKENS = {
    "the", "with", "and", "pour", "avec", "game", "games", "gioco", "giochi", "jeu", "juego",
    "ps4", "playstation", "borsa", "bag", "sac", "purse", "wallet", "portafoglio", "scarf",
    "prada", "gucci", "no", "brand", "uomo", "donna", "woman", "man",
}

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


def get_image_embedder(model_name: str):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "projection_dim", None)
    if output_dim is None:
        raise RuntimeError(f"Could not infer embedding size for image model: {model_name}")

    def embed_images(images, batch_size: int = 8):
        batches = []
        with torch.inference_mode():
            for start in range(0, len(images), max(1, int(batch_size))):
                batch = images[start:start + max(1, int(batch_size))]
                inputs = processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs)
                if getattr(outputs, "pooler_output", None) is not None:
                    vecs = outputs.pooler_output
                elif getattr(outputs, "image_embeds", None) is not None:
                    vecs = outputs.image_embeds
                elif getattr(outputs, "last_hidden_state", None) is not None:
                    vecs = outputs.last_hidden_state[:, 0]
                else:
                    raise RuntimeError(f"Unsupported output structure for image model: {model_name}")
                vecs = F.normalize(vecs, dim=1)
                batches.append(vecs.detach().cpu().numpy().astype(np.float32))
        if not batches:
            return np.zeros((0, int(output_dim)), dtype=np.float32)
        return np.vstack(batches).astype(np.float32)

    return embed_images, f"transformers:{model_name}", int(output_dim)


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 1e-6, norms, 1.0)
    return (X / norms).astype(np.float32)


def combine_feature_blocks(*blocks) -> np.ndarray:
    valid = []
    for block in blocks:
        if block is None:
            continue
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0 or arr.shape[1] == 0:
            continue
        valid.append(arr)
    if not valid:
        raise ValueError("At least one feature block is required")
    if len(valid) == 1:
        return l2_normalize_rows(valid[0])
    return l2_normalize_rows(np.hstack(valid).astype(np.float32))


def select_primary_image_source(row) -> str:
    for key in ("LocalPrimaryImagePath", "LocalImagePaths", "Images"):
        values = normalize_image_sources(row.get(key))
        if values:
            return values[0]
    return ""


def compute_listing_image_embeddings(df: pd.DataFrame, model_name: str, timeout: float = 8.0, batch_size: int = 8):
    primary_sources = [select_primary_image_source(row) for row in df.to_dict("records")]
    embed_images, image_embedder_name, dim = get_image_embedder(model_name)

    source_to_vec: dict[str, np.ndarray] = {}
    unique_sources = [src for src in dict.fromkeys(primary_sources) if src]
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(unique_sources), chunk_size):
        batch_sources = unique_sources[start:start + chunk_size]
        batch_images = []
        loaded_sources = []
        for source in batch_sources:
            try:
                batch_images.append(image_from_source(source, timeout=timeout))
                loaded_sources.append(source)
            except Exception:
                continue
        if not batch_images:
            continue
        batch_vecs = embed_images(batch_images, batch_size=chunk_size)
        for source, vec in zip(loaded_sources, batch_vecs):
            source_to_vec[source] = vec.astype(np.float32)

    image_vectors = np.zeros((len(df), dim), dtype=np.float32)
    has_image_embedding = np.zeros(len(df), dtype=bool)
    for idx, source in enumerate(primary_sources):
        vec = source_to_vec.get(source)
        if vec is None:
            continue
        image_vectors[idx] = vec
        has_image_embedding[idx] = True

    return primary_sources, image_vectors, has_image_embedding, image_embedder_name

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

def variant_cluster(df_prod, embed, core_frac, vthr, pwt, image_vectors=None, image_weight: float = 0.0):
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

    Xv_base = np.hstack([V, (pwt * lp).reshape(-1, 1)]).astype(np.float32)
    if image_vectors is not None and float(image_weight) > 0.0:
        Xv_cluster = combine_feature_blocks(Xv_base, float(image_weight) * np.asarray(image_vectors, dtype=np.float32))
    else:
        Xv_cluster = l2_normalize_rows(Xv_base)
    labs = cluster_agglomerative_cosine(Xv_cluster, distance_threshold=vthr) if len(df_prod) > 1 else np.array([0], dtype=int)
    return vtexts, labs, core, counts, Xv_base, Xv_cluster

def deal_score_variant(prices: np.ndarray, price: float):
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 3 or not np.isfinite(price) or price <= 0:
        return np.nan
    med = np.median(prices)
    mad = np.median(np.abs(prices - med))
    scale = 1.4826 * mad + 1e-6
    return float((med - price) / scale)


def estimate_resale_metrics(prices: np.ndarray, buy_price: float, fee_rate: float, fixed_cost: float, safety_discount: float):
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 3 or not np.isfinite(buy_price) or buy_price <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    q25 = float(np.quantile(prices, 0.25))
    med = float(np.median(prices))
    q75 = float(np.quantile(prices, 0.75))
    conservative_resale = med * (1.0 - safety_discount)
    conservative_resale = min(conservative_resale, q75)
    conservative_resale = max(conservative_resale, q25)
    net_proceeds = conservative_resale * (1.0 - fee_rate) - fixed_cost
    expected_profit = net_proceeds - buy_price
    expected_profit_margin = expected_profit / buy_price if buy_price > 0 else np.nan
    return q25, med, conservative_resale, net_proceeds, expected_profit_margin


def compute_resale_safety_score(deal_score: float, confidence: float, expected_profit: float, expected_profit_margin: float) -> float:
    if not np.isfinite(confidence):
        return np.nan
    score_component = np.clip((deal_score if np.isfinite(deal_score) else 0.0) / 6.0, 0.0, 1.0)
    margin_component = np.clip((expected_profit_margin if np.isfinite(expected_profit_margin) else 0.0) / 0.30, 0.0, 1.0)
    profit_component = np.clip((expected_profit if np.isfinite(expected_profit) else 0.0) / 20.0, 0.0, 1.0)
    raw = 100.0 * confidence * (0.4 * score_component + 0.4 * margin_component + 0.2 * profit_component)
    return float(np.clip(raw, 0.0, 100.0))


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


def variant_mad_ratio(prices: np.ndarray) -> float:
    med, mad = robust_stats(prices)
    return safe_ratio(mad, med)


def dedupe_listing_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    dedupe_subset = ["Dataid"] if "Dataid" in df.columns else ["Link"] if "Link" in df.columns else None
    if not dedupe_subset:
        out = df.copy()
        out["SnapshotCount"] = 1
        return out

    out = df.copy()
    out["SnapshotCount"] = out.groupby(dedupe_subset)[dedupe_subset[0]].transform("size")
    out["_row_order"] = range(len(out))
    sort_cols = []
    temp_cols = ["_row_order"]
    if "SearchCount" in out.columns:
        out["_SearchCountNum"] = pd.to_numeric(out["SearchCount"], errors="coerce")
        sort_cols.append("_SearchCountNum")
        temp_cols.append("_SearchCountNum")
    if "SearchDate" in out.columns:
        out["_SearchDateTs"] = pd.to_datetime(out["SearchDate"], errors="coerce", dayfirst=True)
        sort_cols.append("_SearchDateTs")
        temp_cols.append("_SearchDateTs")
    if "Page" in out.columns:
        out["_PageNum"] = pd.to_numeric(out["Page"], errors="coerce")
        sort_cols.append("_PageNum")
        temp_cols.append("_PageNum")
    sort_cols.append("_row_order")
    out = out.sort_values(sort_cols, kind="stable")
    out = out.drop_duplicates(subset=dedupe_subset, keep="last")
    out = out.drop(columns=temp_cols, errors="ignore")
    return out.reset_index(drop=True)


def title_signal_features(title_norm: str, brand_norm: str = ""):
    toks = tokenize(title_norm)
    brand_toks = set(tokenize(brand_norm))
    informative = [t for t in toks if t not in brand_toks and t not in GENERIC_TITLE_TOKENS and not t.isdigit()]
    informative_unique = sorted(set(informative))
    has_code = int(bool(extract_model_codes(title_norm)))
    is_generic = int(len(informative_unique) <= 1 and not has_code)
    return len(informative_unique), has_code, is_generic

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
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.title(f"Product {product_id} - Variant price distributions")
    plt.xlabel("VariantId"); plt.ylabel("Price")
    out = os.path.join(plots_dir, f"product_{product_id}_variant_prices.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

def plot_variant_pca(Xv, labs, product_id, plots_dir):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    if Xv.shape[0] < 3:
        return
    # Skip PCA when every point is effectively identical; sklearn emits a runtime
    # warning because the total variance is zero in that case.
    if not np.isfinite(Xv).all() or float(np.var(Xv, axis=0).sum()) <= 1e-12:
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
    raw_row_count = len(df)
    df = dedupe_listing_snapshots(df)
    df["Title_norm"] = df["Title"].apply(normalize_text)
    df["Brand_norm"] = df["Brand"].apply(normalize_text) if "Brand" in df.columns else ""
    title_features = df.apply(lambda row: title_signal_features(row["Title_norm"], row.get("Brand_norm", "")), axis=1, result_type="expand")
    title_features.columns = ["InformativeTokenCount", "HasModelCode", "IsGenericTitle"]
    df[["InformativeTokenCount", "HasModelCode", "IsGenericTitle"]] = title_features
    df["NegFlags"] = df["Title_norm"].apply(lambda t: "|".join(negative_flags(t, DEFAULT_NEGATIVE_KEYWORDS)))
    df["HasNegFlag"] = df["NegFlags"].astype(str).apply(lambda s: s not in ("", "nan"))
    df["PrimaryImageSource"] = df.apply(select_primary_image_source, axis=1)
    df["HasImageEmbedding"] = False

    df["BlockKey"] = df.apply(make_block_key, axis=1)
    size_col = df["Size"].astype(str) if "Size" in df.columns else pd.Series([""] * len(df), index=df.index)
    df["EmbedText"] = (df["Title_norm"].fillna("") + " | " + df["Brand_norm"].fillna("") + " | " + size_col.fillna("")).astype(str)

    embed, embedder_name = get_embedder(args.model)
    X = embed(df["EmbedText"].tolist())
    image_vectors = None
    product_cluster_vectors = X
    image_embedder_name = ""
    if getattr(args, "use_image_embeddings", False):
        primary_sources, image_vectors, has_image_embedding, image_embedder_name = compute_listing_image_embeddings(
            df,
            args.image_embedding_model,
            timeout=args.image_embedding_timeout,
            batch_size=args.image_embedding_batch_size,
        )
        df["PrimaryImageSource"] = primary_sources
        df["HasImageEmbedding"] = has_image_embedding.astype(bool)
        if bool(has_image_embedding.any()) and float(args.product_image_weight) > 0.0:
            product_cluster_vectors = combine_feature_blocks(X, float(args.product_image_weight) * image_vectors)
        print(f"Image embeddings ready for {int(has_image_embedding.sum())}/{len(df)} listings using {image_embedder_name}")

    product_id = np.full(len(df), -1, dtype=int)
    product_meta = {}
    next_pid = 0

    for block_key, idxs in df.groupby("BlockKey").indices.items():
        idxs = list(idxs)
        if len(idxs) == 1:
            pid = next_pid
            next_pid += 1
            product_id[idxs[0]] = pid
            product_meta[pid] = (block_key, choose_canonical_name([df.loc[idxs[0], "Title_norm"]]), int(idxs[0]), 1)
            continue

        labs = cluster_agglomerative_cosine(product_cluster_vectors[idxs], distance_threshold=args.product_threshold)
        for lab in np.unique(labs):
            members = [idxs[i] for i in range(len(idxs)) if labs[i] == lab]
            pid = next_pid
            next_pid += 1
            for m in members:
                product_id[m] = pid
            rep_i = representative_index(X, members)
            cname = choose_canonical_name(df.loc[members, "Title_norm"].tolist())
            product_meta[pid] = (block_key, cname, int(rep_i), len(members))

    df["ProductId"] = product_id

    next_vid = 0
    df_parts = []
    variant_rows = []
    trust_rows = []

    for pid, g in df.groupby("ProductId"):
        g = g.copy()
        block_key, cname, rep_i, nprod = product_meta[int(pid)]
        idxs = g.index.to_list()
        centroid = X[idxs].mean(axis=0)
        core_counts = build_core_token_counts(g["Title_norm"].tolist())

        if args.autotune_variants and len(g) >= args.min_product_size_for_variants:
            core_frac, vthr, pwt = autotune_variant_params(g)
        else:
            core_frac, vthr, pwt = args.core_frac, args.variant_threshold, args.variant_price_weight

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

        if len(g) < args.min_product_size_for_variants:
            vtexts = [""] * len(g)
            vlabs = np.zeros(len(g), dtype=int)
            Xv_base = np.zeros((len(g), X.shape[1] + 1), dtype=np.float32)
            Xv_cluster = Xv_base
        else:
            variant_image_vectors = image_vectors[idxs] if image_vectors is not None else None
            vtexts, vlabs, _, _, Xv_base, Xv_cluster = variant_cluster(
                g,
                embed,
                core_frac,
                vthr,
                pwt,
                image_vectors=variant_image_vectors,
                image_weight=float(getattr(args, "variant_image_weight", 0.0)),
            )

        g["VariantText"] = vtexts
        g["VariantId_local"] = vlabs

        local_to_global = {}
        for lab in sorted(np.unique(vlabs)):
            local_to_global[int(lab)] = next_vid
            next_vid += 1
        g["VariantId"] = g["VariantId_local"].map(local_to_global).astype(int)
        g["VariantCentroidSim"] = np.nan
        g["VariantCount"] = 0
        g["VariantPriceQ25"] = np.nan
        g["VariantPriceMedian"] = np.nan
        g["VariantPriceQ75"] = np.nan
        g["VariantPriceMAD"] = np.nan
        g["VariantPriceMADRatio"] = np.nan

        try:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(Xv_cluster, vlabs, metric="cosine")) if len(np.unique(vlabs)) >= 2 else np.nan
        except Exception:
            sil = np.nan
        g["ProductVariantSilhouette"] = sil

        variant_sizes = []
        variant_mad_ratios = []
        for lab in sorted(np.unique(vlabs)):
            member_positions = np.flatnonzero(vlabs == lab)
            members = g.index[member_positions].tolist()
            vid = local_to_global[int(lab)]
            member_vecs = Xv_cluster[member_positions]
            vcentroid = member_vecs.mean(axis=0) if len(member_positions) else np.zeros(Xv_cluster.shape[1], dtype=np.float32)
            norms = np.linalg.norm(member_vecs, axis=1) * max(np.linalg.norm(vcentroid), 1e-6)
            numerators = member_vecs @ vcentroid
            sims = np.full(len(member_positions), np.nan, dtype=np.float32)
            np.divide(numerators, norms, out=sims, where=norms > 0)
            g.loc[members, "VariantCentroidSim"] = sims
            g.loc[members, "VariantCount"] = int(len(members))

            prices = pd.to_numeric(g.loc[members, "Price"], errors="coerce").to_numpy(dtype=np.float32)
            med, mad = robust_stats(prices)
            mad_ratio = variant_mad_ratio(prices)
            valid_prices = prices[np.isfinite(prices) & (prices > 0)]
            q25 = float(np.quantile(valid_prices, 0.25)) if valid_prices.size else np.nan
            q75 = float(np.quantile(valid_prices, 0.75)) if valid_prices.size else np.nan
            g.loc[members, "VariantPriceQ25"] = q25
            g.loc[members, "VariantPriceMedian"] = med
            g.loc[members, "VariantPriceQ75"] = q75
            g.loc[members, "VariantPriceMAD"] = mad
            g.loc[members, "VariantPriceMADRatio"] = mad_ratio
            variant_sizes.append(int(len(members)))
            if np.isfinite(mad_ratio):
                variant_mad_ratios.append(float(mad_ratio))

            buf = [float(x) for x in prices[np.isfinite(prices) & (prices > 0)][-args.price_buffer_size:]]
            vtoks = []
            for s in g.loc[members, "VariantText"].fillna("").tolist():
                vtoks.extend(tokenize(s))
            top_v = " ".join([w for w, _ in Counter(vtoks).most_common(8)])

            persist_member_vecs = Xv_base[member_positions]
            persist_centroid = persist_member_vecs.mean(axis=0) if len(member_positions) else np.zeros(Xv_base.shape[1], dtype=np.float32)
            score.upsert_variant(con, int(vid), int(pid), persist_centroid, int(len(members)), float(vthr), float(pwt), float(core_frac), buf, top_v)

            variant_rows.append({
                "ProductId": int(pid),
                "VariantId": int(vid),
                "VariantCount": int(len(members)),
                "VariantTextTop": top_v,
                "PriceQ25": q25,
                "PriceMedian": med,
                "PriceQ75": q75,
                "PriceMAD": mad,
                "PriceMADRatio": mad_ratio,
                "PriceMin": float(np.nanmin(prices)) if np.isfinite(prices).any() else np.nan,
                "PriceMax": float(np.nanmax(prices)) if np.isfinite(prices).any() else np.nan,
                "CoreFrac": float(core_frac),
                "VariantThreshold": float(vthr),
                "PriceWeight": float(pwt),
            })

        trust_rows.append({
            "ProductId": int(pid),
            "n": int(len(g)),
            "VariantsFound": int(len(np.unique(vlabs))),
            "VariantSilhouette": sil,
            "GenericTitleShare": float(g["IsGenericTitle"].mean()) if len(g) else np.nan,
            "MedianInformativeTokens": float(g["InformativeTokenCount"].median()) if len(g) else np.nan,
            "VariantSizeMin": int(min(variant_sizes)) if variant_sizes else 0,
            "VariantSizeMax": int(max(variant_sizes)) if variant_sizes else 0,
            "VariantPriceMADRatioMedian": float(np.median(variant_mad_ratios)) if variant_mad_ratios else np.nan,
        })

        if args.make_plots:
            plot_variant_prices(g, int(pid), plots_dir)
            plot_variant_pca(Xv_cluster, vlabs, int(pid), plots_dir)

        df_parts.append(g)

    df2 = pd.concat(df_parts, ignore_index=True)
    df2["_PriceNum"] = pd.to_numeric(df2.get("Price"), errors="coerce")

    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    prices_by_variant = {int(vid): grp["_PriceNum"].to_numpy(dtype=np.float32) for vid, grp in df2.groupby("VariantId")}

    scores_raw, scores_final, confidences, eligible_flags, notes = [], [], [], [], []
    expected_resales, expected_net_proceeds, expected_profits, expected_profit_margins, resale_safety_scores = [], [], [], [], []
    for _, row in df2.iterrows():
        vid = int(row["VariantId"])
        price = float(row["_PriceNum"]) if np.isfinite(row["_PriceNum"]) else np.nan
        z = deal_score_variant(prices_by_variant.get(vid, np.array([])), price)
        penalty = 0.0
        confidence_penalty = 0.0
        nn = []
        hard_fail = False

        if bool(row.get("HasNegFlag", False)):
            penalty += 0.6
            confidence_penalty += 0.20
            nn.append("negflag")
            if args.exclude_negflag_deals:
                hard_fail = True

        if int(row.get("IsGenericTitle", 0)):
            penalty += 1.1
            confidence_penalty += 0.35
            nn.append("generic_title")
            if int(row.get("HasModelCode", 0)) == 0 and int(row.get("InformativeTokenCount", 0)) < args.min_informative_tokens:
                hard_fail = True

        if int(row.get("InformativeTokenCount", 0)) < args.min_informative_tokens and int(row.get("HasModelCode", 0)) == 0:
            penalty += 0.5
            confidence_penalty += 0.15
            nn.append("low_title_specificity")

        snapshot_count = int(row.get("SnapshotCount", 1) or 1)
        if snapshot_count >= 3:
            confidence_penalty += min(0.25, 0.05 * (snapshot_count - 2))
            nn.append("lingered_across_snapshots")

        variant_count = int(row.get("VariantCount", 0) or 0)
        if variant_count < args.min_variant_size_for_confident_deals:
            confidence_penalty += 0.35
            nn.append("small_variant")
            hard_fail = True

        mad_ratio = float(row.get("VariantPriceMADRatio", np.nan)) if pd.notna(row.get("VariantPriceMADRatio", np.nan)) else np.nan
        if np.isfinite(mad_ratio) and mad_ratio > args.max_variant_mad_ratio:
            penalty += min(1.0, (mad_ratio - args.max_variant_mad_ratio) * 4.0)
            confidence_penalty += min(0.35, (mad_ratio - args.max_variant_mad_ratio) * 1.5)
            nn.append("wide_price_dispersion")
        if np.isfinite(mad_ratio) and mad_ratio > args.hard_max_variant_mad_ratio:
            hard_fail = True
            nn.append("very_wide_price_dispersion")

        sil = float(row.get("ProductVariantSilhouette", np.nan)) if pd.notna(row.get("ProductVariantSilhouette", np.nan)) else np.nan
        if np.isfinite(sil) and sil < args.min_variant_silhouette:
            penalty += min(0.8, (args.min_variant_silhouette - sil) * 2.0)
            confidence_penalty += min(0.30, (args.min_variant_silhouette - sil) * 1.2)
            nn.append("low_variant_silhouette")
            hard_fail = True

        sim = float(row.get("VariantCentroidSim", np.nan)) if pd.notna(row.get("VariantCentroidSim", np.nan)) else np.nan
        if np.isfinite(sim) and sim < args.min_centroid_similarity:
            penalty += min(0.8, (args.min_centroid_similarity - sim) * 2.0)
            confidence_penalty += min(0.25, (args.min_centroid_similarity - sim) * 1.2)
            nn.append("off_centroid")

        v_med = float(row.get("VariantPriceMedian", np.nan)) if pd.notna(row.get("VariantPriceMedian", np.nan)) else np.nan
        if np.isfinite(price) and np.isfinite(v_med) and v_med > 0 and price < 0.45 * v_med:
            penalty += 0.5
            confidence_penalty += 0.10
            nn.append("very_low_vs_variant_median")

        _, _, expected_resale, expected_net, expected_margin = estimate_resale_metrics(
            prices_by_variant.get(vid, np.array([])),
            price,
            args.resale_fee_rate,
            args.resale_fixed_cost,
            args.resale_safety_discount,
        )
        expected_profit = expected_net - price if np.isfinite(expected_net) and np.isfinite(price) else np.nan
        if np.isfinite(expected_margin) and expected_margin < args.min_expected_profit_margin:
            confidence_penalty += 0.20
            nn.append("low_expected_profit_margin")
            hard_fail = True
        if np.isfinite(expected_profit) and expected_profit < args.min_expected_profit:
            confidence_penalty += 0.20
            nn.append("low_expected_profit")
            hard_fail = True

        confidence = clamp(1.0 - confidence_penalty, 0.0, 1.0)
        conservative = clamp((z - penalty) * confidence, -10, 10) if np.isfinite(z) else np.nan
        eligible = np.isfinite(conservative) and (confidence >= args.min_deal_confidence) and (not hard_fail)
        if not eligible:
            conservative = np.nan
            nn.append("filtered_low_confidence")

        scores_raw.append(z)
        scores_final.append(conservative)
        confidences.append(confidence)
        eligible_flags.append(bool(eligible))
        notes.append("|".join(dict.fromkeys(nn)))
        expected_resales.append(expected_resale)
        expected_net_proceeds.append(expected_net)
        expected_profits.append(expected_profit)
        expected_profit_margins.append(expected_margin)
        resale_safety_scores.append(compute_resale_safety_score(conservative, confidence, expected_profit, expected_margin))

    df2["DealScoreRaw"] = scores_raw
    df2["DealConfidence"] = confidences
    df2["DealEligible"] = eligible_flags
    df2["DealScore"] = scores_final
    df2["DealNotes"] = notes
    df2["ExpectedResalePrice"] = expected_resales
    df2["ExpectedNetProceeds"] = expected_net_proceeds
    df2["ExpectedProfit"] = expected_profits
    df2["ExpectedProfitMargin"] = expected_profit_margins
    df2["ResaleSafetyScore"] = resale_safety_scores
    df2["VariantClusterSize"] = df2["VariantCount"].astype(int)

    items_out = os.path.join(args.out_dir, "items_with_product_and_variant.csv")
    var_out = os.path.join(args.out_dir, "variant_summary.csv")
    trust_out = os.path.join(args.out_dir, "trust_report.csv")
    deals_out = os.path.join(args.out_dir, "deals_ranked.csv")

    df2.to_csv(items_out, index=False)
    pd.DataFrame(variant_rows).to_csv(var_out, index=False)
    pd.DataFrame(trust_rows).to_csv(trust_out, index=False)

    deals_ranked = df2.copy()
    deals_ranked = deals_ranked[deals_ranked["VariantClusterSize"] >= args.min_variant_size_for_deals]
    deals_ranked = deals_ranked[deals_ranked["DealEligible"] & deals_ranked["DealScore"].notna()]
    deals_ranked = deals_ranked.sort_values(["ResaleSafetyScore", "ExpectedProfitMargin", "DealScore", "DealConfidence", "SnapshotCount"], ascending=[False, False, False, False, True])
    deals_ranked.to_csv(deals_out, index=False)

    score.set_meta(con, "embedder", embedder_name)
    score.set_meta(con, "model_name", args.model)
    score.set_meta(con, "use_image_embeddings", bool(getattr(args, "use_image_embeddings", False)))
    score.set_meta(con, "image_embedder", image_embedder_name)

    print("✅ Batch complete")
    print(f"- Input rows:       {raw_row_count}")
    print(f"- Unique listings:  {len(df)}")
    print(f"- Index DB:         {args.db}")
    print(f"- Items:            {items_out}")
    print(f"- Deals:            {deals_out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="./out")
    ap.add_argument("--db", default="./out/index.sqlite")

    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--product_threshold", type=float, default=0.24)
    ap.add_argument("--use_image_embeddings", action="store_true")
    ap.add_argument("--image_embedding_model", default="facebook/dinov2-base")
    ap.add_argument("--image_embedding_timeout", type=float, default=8.0)
    ap.add_argument("--image_embedding_batch_size", type=int, default=8)
    ap.add_argument("--product_image_weight", type=float, default=0.15)

    ap.add_argument("--autotune_variants", action="store_true")
    ap.add_argument("--variant_threshold", type=float, default=0.33)
    ap.add_argument("--core_frac", type=float, default=0.70)
    ap.add_argument("--variant_price_weight", type=float, default=0.35)
    ap.add_argument("--variant_image_weight", type=float, default=0.20)

    ap.add_argument("--min_product_size_for_variants", type=int, default=4)
    # Recall-oriented defaults: keep more clustered listings alive for downstream ranking.
    ap.add_argument("--min_variant_size_for_deals", type=int, default=3)
    ap.add_argument("--min_variant_size_for_confident_deals", type=int, default=3)
    ap.add_argument("--min_variant_silhouette", type=float, default=0.15)
    ap.add_argument("--max_variant_mad_ratio", type=float, default=0.35)
    ap.add_argument("--hard_max_variant_mad_ratio", type=float, default=0.60)
    ap.add_argument("--min_deal_confidence", type=float, default=0.35)
    ap.add_argument("--min_centroid_similarity", type=float, default=0.45)
    ap.add_argument("--min_informative_tokens", type=int, default=2)
    ap.add_argument("--exclude_negflag_deals", action="store_true")
    ap.add_argument("--resale_fee_rate", type=float, default=0.10)
    ap.add_argument("--resale_fixed_cost", type=float, default=0.0)
    ap.add_argument("--resale_safety_discount", type=float, default=0.05)
    ap.add_argument("--min_expected_profit", type=float, default=0.0)
    ap.add_argument("--min_expected_profit_margin", type=float, default=0.0)

    ap.add_argument("--price_buffer_size", type=int, default=200)

    ap.add_argument("--make_plots", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    run(parse_args())
