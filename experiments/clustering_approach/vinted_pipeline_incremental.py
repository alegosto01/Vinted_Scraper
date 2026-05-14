#!/usr/bin/env python3
"""
vinted_pipeline_incremental.py

Incremental mode:
- loads SQLite index (products + variants)
- reads ONLY new items (either from a small "new_items.csv" or from a growing CSV with last_seen_rows)
- assigns ProductId + VariantId (stable)
- updates centroids + variant price buffers
- computes conservative DealScore / DealConfidence for new items only
- appends results to an output CSV (or writes a new one)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiments.clustering_approach.vinted_index_score as store

ALIAS_MAP = {
    r"\baf\s?1\b": "air force 1",
    r"\baf1\b": "air force 1",
    r"\blv\b": "louis vuitton",
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MODEL_CODE_RE = re.compile(r"\b[a-z]{1,4}\d{2,6}\b|\b\d{2,4}\b", re.IGNORECASE)
DEFAULT_NEGATIVE_KEYWORDS = [
    "replica", "fake", "falso", "tarocco", "ispirato", "inspired", "like", "stile",
    "lotto", "bundle", "stock", "solo scatola", "scatola", "dustbag", "ricambio", "solo",
    "rovinato", "strappato", "macchia", "difetto", "da riparare", "rott",
    "kids", "bambino", "bimbi", "ragazzo", "ragazza",
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


def extract_model_codes(title_norm: str, max_codes: int = 5):
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


def negative_flags(title_norm: str, neg_keywords):
    return [kw for kw in neg_keywords if kw and kw in title_norm]


def title_signal_features(title_norm: str, brand_norm: str = ""):
    toks = tokenize(title_norm)
    brand_toks = set(tokenize(brand_norm))
    informative = [t for t in toks if t not in brand_toks and t not in GENERIC_TITLE_TOKENS and not t.isdigit()]
    informative_unique = sorted(set(informative))
    has_code = int(bool(extract_model_codes(title_norm)))
    is_generic = int(len(informative_unique) <= 1 and not has_code)
    return len(informative_unique), has_code, is_generic


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


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


def variant_mad_ratio(prices: np.ndarray) -> float:
    med, mad = robust_stats(prices)
    return safe_ratio(mad, med)


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


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def embed(texts):
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    return embed


def nearest_centroid(centroids: np.ndarray, x: np.ndarray):
    sims = centroids @ x
    j = int(np.argmax(sims))
    dist = float(1.0 - sims[j])
    return j, dist


def online_mean_update(mu: np.ndarray, n: int, x: np.ndarray):
    return mu + (x - mu) / float(n + 1)


def top_tokens(texts, k=8):
    toks = []
    for s in texts:
        toks.extend(tokenize(s))
    return " ".join([w for w, _ in Counter(toks).most_common(k)])


def preprocess_new_rows(df_new: pd.DataFrame) -> pd.DataFrame:
    df_new = df_new.copy()
    df_new["Title_norm"] = df_new["Title"].apply(normalize_text)
    df_new["Brand_norm"] = df_new["Brand"].apply(normalize_text) if "Brand" in df_new.columns else ""
    size_col = df_new["Size"].astype(str) if "Size" in df_new.columns else pd.Series([""] * len(df_new), index=df_new.index)
    df_new["EmbedText"] = (df_new["Title_norm"].fillna("") + " | " + df_new["Brand_norm"].fillna("") + " | " + size_col.fillna("")).astype(str)
    title_features = df_new.apply(lambda row: title_signal_features(row["Title_norm"], row.get("Brand_norm", "")), axis=1, result_type="expand")
    title_features.columns = ["InformativeTokenCount", "HasModelCode", "IsGenericTitle"]
    df_new[["InformativeTokenCount", "HasModelCode", "IsGenericTitle"]] = title_features
    if "NegFlags" not in df_new.columns:
        df_new["NegFlags"] = df_new["Title_norm"].apply(lambda t: "|".join(negative_flags(t, DEFAULT_NEGATIVE_KEYWORDS)))
    if "HasNegFlag" not in df_new.columns:
        df_new["HasNegFlag"] = df_new["NegFlags"].astype(str).apply(lambda s: s not in ("", "nan"))
    return df_new


def conservative_score_row(row, arr, z, config):
    price = float(row.get("_PriceNum", np.nan)) if pd.notna(row.get("_PriceNum", np.nan)) else np.nan
    penalty = 0.0
    confidence_penalty = 0.0
    notes = []
    hard_fail = False

    if bool(row.get("HasNegFlag", False)):
        penalty += 0.6
        confidence_penalty += 0.20
        notes.append("negflag")
        if config["exclude_negflag_deals"]:
            hard_fail = True

    if int(row.get("IsGenericTitle", 0)):
        penalty += 1.1
        confidence_penalty += 0.35
        notes.append("generic_title")
        if int(row.get("HasModelCode", 0)) == 0 and int(row.get("InformativeTokenCount", 0)) < config["min_informative_tokens"]:
            hard_fail = True

    if int(row.get("InformativeTokenCount", 0)) < config["min_informative_tokens"] and int(row.get("HasModelCode", 0)) == 0:
        penalty += 0.5
        confidence_penalty += 0.15
        notes.append("low_title_specificity")

    snapshot_count = int(row.get("SnapshotCount", 1) or 1)
    if snapshot_count >= 3:
        confidence_penalty += min(0.25, 0.05 * (snapshot_count - 2))
        notes.append("lingered_across_snapshots")

    variant_count = int(row.get("VariantCount", 0) or 0)
    if variant_count < config["min_variant_size_for_confident_deals"]:
        confidence_penalty += 0.35
        notes.append("small_variant")
        hard_fail = True

    mad_ratio = float(row.get("VariantPriceMADRatio", np.nan)) if pd.notna(row.get("VariantPriceMADRatio", np.nan)) else np.nan
    if np.isfinite(mad_ratio) and mad_ratio > config["max_variant_mad_ratio"]:
        penalty += min(1.0, (mad_ratio - config["max_variant_mad_ratio"]) * 4.0)
        confidence_penalty += min(0.35, (mad_ratio - config["max_variant_mad_ratio"]) * 1.5)
        notes.append("wide_price_dispersion")
    if np.isfinite(mad_ratio) and mad_ratio > config["hard_max_variant_mad_ratio"]:
        hard_fail = True
        notes.append("very_wide_price_dispersion")

    sim = float(row.get("VariantCentroidSim", np.nan)) if pd.notna(row.get("VariantCentroidSim", np.nan)) else np.nan
    if np.isfinite(sim) and sim < config["min_centroid_similarity"]:
        penalty += min(0.8, (config["min_centroid_similarity"] - sim) * 2.0)
        confidence_penalty += min(0.25, (config["min_centroid_similarity"] - sim) * 1.2)
        notes.append("off_centroid")

    v_med, _ = robust_stats(arr)
    if np.isfinite(price) and np.isfinite(v_med) and v_med > 0 and price < 0.45 * v_med:
        penalty += 0.5
        confidence_penalty += 0.10
        notes.append("very_low_vs_variant_median")

    _, _, expected_resale, expected_net, expected_margin = estimate_resale_metrics(
        arr,
        price,
        config["resale_fee_rate"],
        config["resale_fixed_cost"],
        config["resale_safety_discount"],
    )
    expected_profit = expected_net - price if np.isfinite(expected_net) and np.isfinite(price) else np.nan
    if np.isfinite(expected_margin) and expected_margin < config["min_expected_profit_margin"]:
        confidence_penalty += 0.20
        notes.append("low_expected_profit_margin")
        hard_fail = True
    if np.isfinite(expected_profit) and expected_profit < config["min_expected_profit"]:
        confidence_penalty += 0.20
        notes.append("low_expected_profit")
        hard_fail = True

    confidence = clamp(1.0 - confidence_penalty, 0.0, 1.0)
    conservative = clamp((z - penalty) * confidence, -10, 10) if np.isfinite(z) else np.nan
    eligible = np.isfinite(conservative) and (confidence >= config["min_deal_confidence"]) and (not hard_fail)
    if not eligible:
        conservative = np.nan
        notes.append("filtered_low_confidence")
    return conservative, confidence, eligible, notes, expected_resale, expected_net, expected_profit, expected_margin


def process_new_df(
    df_new: pd.DataFrame,
    db_path: str,
    price_buffer_size: int = 200,
    min_variant_size_for_confident_deals: int = 8,
    max_variant_mad_ratio: float = 0.25,
    hard_max_variant_mad_ratio: float = 0.45,
    min_deal_confidence: float = 0.55,
    min_centroid_similarity: float = 0.55,
    min_informative_tokens: int = 2,
    exclude_negflag_deals: bool = False,
    resale_fee_rate: float = 0.10,
    resale_fixed_cost: float = 0.0,
    resale_safety_discount: float = 0.05,
    min_expected_profit: float = 0.0,
    min_expected_profit_margin: float = 0.0,
):
    con = store.connect(db_path)
    model_name = store.get_meta(con, "model_name", "paraphrase-multilingual-MiniLM-L12-v2")
    embed = get_embedder(model_name)

    products = store.load_products(con)
    variants = store.load_variants(con)
    _, next_vid = store.get_next_ids(con)

    if df_new.empty:
        return df_new

    df_new = preprocess_new_rows(df_new)
    df_new["_PriceNum"] = pd.to_numeric(df_new.get("Price"), errors="coerce")

    if not products:
        raise RuntimeError("Index has no products. Run batch first.")

    pids = sorted(products.keys())
    P = np.stack([products[pid]["centroid"] for pid in pids], axis=0).astype(np.float32)
    X = embed(df_new["EmbedText"].tolist())

    assigned_pids = []
    assigned_vids = []
    variant_texts = []
    deal_scores_raw = []
    deal_scores = []
    deal_confidences = []
    deal_eligible = []
    deal_notes = []
    variant_counts = []
    variant_centroid_sims = []
    variant_mads = []
    variant_mad_ratios = []
    variant_price_q25s = []
    variant_price_medians = []
    expected_resales = []
    expected_net_proceeds = []
    expected_profits = []
    expected_profit_margins = []
    resale_safety_scores = []

    config = {
        "min_variant_size_for_confident_deals": int(min_variant_size_for_confident_deals),
        "max_variant_mad_ratio": float(max_variant_mad_ratio),
        "hard_max_variant_mad_ratio": float(hard_max_variant_mad_ratio),
        "min_deal_confidence": float(min_deal_confidence),
        "min_centroid_similarity": float(min_centroid_similarity),
        "min_informative_tokens": int(min_informative_tokens),
        "exclude_negflag_deals": bool(exclude_negflag_deals),
        "resale_fee_rate": float(resale_fee_rate),
        "resale_fixed_cost": float(resale_fixed_cost),
        "resale_safety_discount": float(resale_safety_discount),
        "min_expected_profit": float(min_expected_profit),
        "min_expected_profit_margin": float(min_expected_profit_margin),
    }

    for r_i, row in df_new.iterrows():
        x = X[df_new.index.get_loc(r_i)]
        j, dist = nearest_centroid(P, x)
        pid = pids[j]
        thr = products[pid]["product_threshold"]

        if dist > thr:
            assigned_pids.append(-1)
            assigned_vids.append(-1)
            variant_texts.append("")
            deal_scores_raw.append(np.nan)
            deal_scores.append(np.nan)
            deal_confidences.append(0.0)
            deal_eligible.append(False)
            deal_notes.append("unassigned_product")
            variant_counts.append(0)
            variant_centroid_sims.append(np.nan)
            variant_mads.append(np.nan)
            variant_mad_ratios.append(np.nan)
            variant_price_q25s.append(np.nan)
            variant_price_medians.append(np.nan)
            expected_resales.append(np.nan)
            expected_net_proceeds.append(np.nan)
            expected_profits.append(np.nan)
            expected_profit_margins.append(np.nan)
            resale_safety_scores.append(np.nan)
            continue

        mu = products[pid]["centroid"]
        n = products[pid]["n"]
        new_mu = online_mean_update(mu, n, x)
        products[pid]["centroid"] = new_mu
        products[pid]["n"] = n + 1
        store.upsert_product(
            con, pid, new_mu, products[pid]["n"], thr,
            products[pid]["canonical_name"], products[pid]["block_key_hint"],
            products[pid]["core_token_counts"], products[pid]["core_token_min_frac"],
        )
        assigned_pids.append(pid)

        core_counts = products[pid]["core_token_counts"]
        core_frac = products[pid]["core_token_min_frac"]
        core = core_tokens_from_counts(core_counts, products[pid]["n"], core_frac)
        vtext = make_variant_text(row["Title_norm"], core)
        variant_texts.append(vtext)

        vemb = embed([vtext])[0]
        price = float(row["_PriceNum"]) if np.isfinite(row["_PriceNum"]) else np.nan
        vdict = variants.get(pid, {})

        if not vdict:
            vid = next_vid
            next_vid += 1
            price_dim = 0.0
            Vc = np.hstack([vemb, [price_dim]]).astype(np.float32)
            variants.setdefault(pid, {})[vid] = {
                "variant_id": vid,
                "product_id": pid,
                "centroid": Vc,
                "n": 1,
                "variant_threshold": 0.33,
                "price_weight": 0.35,
                "core_frac": core_frac,
                "price_buffer": [price] if np.isfinite(price) else [],
                "variant_text_top": top_tokens([vtext]),
            }
            store.upsert_variant(con, vid, pid, Vc, 1, 0.33, 0.35, core_frac, variants[pid][vid]["price_buffer"], variants[pid][vid]["variant_text_top"])
            assigned_vids.append(vid)
            deal_scores_raw.append(np.nan)
            deal_scores.append(np.nan)
            deal_confidences.append(0.0)
            deal_eligible.append(False)
            deal_notes.append("new_variant_created")
            variant_counts.append(1)
            variant_centroid_sims.append(np.nan)
            variant_mads.append(np.nan)
            variant_mad_ratios.append(np.nan)
            variant_price_q25s.append(np.nan)
            variant_price_medians.append(np.nan)
            expected_resales.append(np.nan)
            expected_net_proceeds.append(np.nan)
            expected_profits.append(np.nan)
            expected_profit_margins.append(np.nan)
            resale_safety_scores.append(np.nan)
            continue

        vids = sorted(vdict.keys())
        Vmat = np.stack([vdict[v]["centroid"] for v in vids], axis=0).astype(np.float32)
        vthr = vdict[vids[0]]["variant_threshold"]
        pwt = vdict[vids[0]]["price_weight"]
        price_dim = float(pwt * np.log(price)) if np.isfinite(price) and price > 0 else 0.0
        xv = np.hstack([vemb, [price_dim]]).astype(np.float32)

        k, vdist = nearest_centroid(Vmat, xv)
        chosen_vid = vids[k]
        if vdist > vthr:
            chosen_vid = next_vid
            next_vid += 1
            Vc = xv.copy()
            vdict[chosen_vid] = {
                "variant_id": chosen_vid,
                "product_id": pid,
                "centroid": Vc,
                "n": 1,
                "variant_threshold": vthr,
                "price_weight": pwt,
                "core_frac": core_frac,
                "price_buffer": [price] if np.isfinite(price) else [],
                "variant_text_top": top_tokens([vtext]),
            }
            store.upsert_variant(con, chosen_vid, pid, Vc, 1, vthr, pwt, core_frac, vdict[chosen_vid]["price_buffer"], vdict[chosen_vid]["variant_text_top"])
            assigned_vids.append(chosen_vid)
            deal_scores_raw.append(np.nan)
            deal_scores.append(np.nan)
            deal_confidences.append(0.0)
            deal_eligible.append(False)
            deal_notes.append("new_variant_created")
            variant_counts.append(1)
            variant_centroid_sims.append(np.nan)
            variant_mads.append(np.nan)
            variant_mad_ratios.append(np.nan)
            variant_price_q25s.append(np.nan)
            variant_price_medians.append(np.nan)
            expected_resales.append(np.nan)
            expected_net_proceeds.append(np.nan)
            expected_profits.append(np.nan)
            expected_profit_margins.append(np.nan)
            resale_safety_scores.append(np.nan)
            continue

        assigned_vids.append(chosen_vid)
        vinfo = vdict[chosen_vid]

        prev_centroid = vinfo["centroid"]
        norms = np.linalg.norm(prev_centroid) * max(np.linalg.norm(xv), 1e-6)
        centroid_sim = float((prev_centroid @ xv) / norms) if norms > 0 else np.nan

        vinfo["centroid"] = online_mean_update(vinfo["centroid"], vinfo["n"], xv)
        vinfo["n"] += 1
        if np.isfinite(price) and price > 0:
            vinfo["price_buffer"].append(price)
            if len(vinfo["price_buffer"]) > price_buffer_size:
                vinfo["price_buffer"] = vinfo["price_buffer"][-price_buffer_size:]
        vinfo["variant_text_top"] = top_tokens([vinfo["variant_text_top"], vtext])
        store.upsert_variant(
            con, chosen_vid, pid, vinfo["centroid"], vinfo["n"],
            vinfo["variant_threshold"], vinfo["price_weight"], vinfo["core_frac"],
            vinfo["price_buffer"], vinfo["variant_text_top"],
        )

        arr = np.array(vinfo["price_buffer"], dtype=np.float32)
        z = deal_score_variant(arr, price) if np.isfinite(price) else np.nan
        q25, med, _, _, _ = estimate_resale_metrics(
            arr,
            price,
            config["resale_fee_rate"],
            config["resale_fixed_cost"],
            config["resale_safety_discount"],
        )
        _, mad = robust_stats(arr)
        mad_ratio = variant_mad_ratio(arr)
        enriched_row = row.to_dict()
        enriched_row.update({
            "_PriceNum": price,
            "VariantCount": int(vinfo["n"]),
            "VariantCentroidSim": centroid_sim,
            "VariantPriceQ25": q25,
            "VariantPriceMedian": med,
            "VariantPriceMAD": mad,
            "VariantPriceMADRatio": mad_ratio,
        })
        score_final, confidence, eligible, notes, expected_resale, expected_net, expected_profit, expected_margin = conservative_score_row(enriched_row, arr, z, config)

        deal_scores_raw.append(z)
        deal_scores.append(score_final)
        deal_confidences.append(confidence)
        deal_eligible.append(eligible)
        deal_notes.append("|".join(dict.fromkeys(notes)))
        variant_counts.append(int(vinfo["n"]))
        variant_centroid_sims.append(centroid_sim)
        variant_mads.append(mad)
        variant_mad_ratios.append(mad_ratio)
        variant_price_q25s.append(q25)
        variant_price_medians.append(med)
        expected_resales.append(expected_resale)
        expected_net_proceeds.append(expected_net)
        expected_profits.append(expected_profit)
        expected_profit_margins.append(expected_margin)
        resale_safety_scores.append(compute_resale_safety_score(score_final, confidence, expected_profit, expected_margin))

    df_new["ProductId"] = assigned_pids
    df_new["VariantId"] = assigned_vids
    df_new["VariantText"] = variant_texts
    df_new["DealScoreRaw"] = deal_scores_raw
    df_new["DealConfidence"] = deal_confidences
    df_new["DealEligible"] = deal_eligible
    df_new["DealScore"] = deal_scores
    df_new["DealNotes"] = deal_notes
    df_new["VariantClusterSize"] = variant_counts
    df_new["VariantCount"] = variant_counts
    df_new["VariantCentroidSim"] = variant_centroid_sims
    df_new["VariantPriceQ25"] = variant_price_q25s
    df_new["VariantPriceMedian"] = variant_price_medians
    df_new["VariantPriceMAD"] = variant_mads
    df_new["VariantPriceMADRatio"] = variant_mad_ratios
    df_new["ExpectedResalePrice"] = expected_resales
    df_new["ExpectedNetProceeds"] = expected_net_proceeds
    df_new["ExpectedProfit"] = expected_profits
    df_new["ExpectedProfitMargin"] = expected_profit_margins
    df_new["ResaleSafetyScore"] = resale_safety_scores
    return df_new


def run(args):
    con = store.connect(args.db)
    if args.new_items:
        df_new = pd.read_csv(args.new_items)
    else:
        last_seen = store.get_processed_state(con, args.growing_csv)
        df_all = pd.read_csv(args.growing_csv)
        df_new = df_all.iloc[last_seen:].copy()
        store.update_processed_state(con, args.growing_csv, len(df_all))

    if df_new.empty:
        print("No new rows to process.")
        return

    df_new = process_new_df(
        df_new,
        db_path=args.db,
        price_buffer_size=args.price_buffer_size,
        min_variant_size_for_confident_deals=args.min_variant_size_for_confident_deals,
        max_variant_mad_ratio=args.max_variant_mad_ratio,
        hard_max_variant_mad_ratio=args.hard_max_variant_mad_ratio,
        min_deal_confidence=args.min_deal_confidence,
        min_centroid_similarity=args.min_centroid_similarity,
        min_informative_tokens=args.min_informative_tokens,
        exclude_negflag_deals=args.exclude_negflag_deals,
        resale_fee_rate=args.resale_fee_rate,
        resale_fixed_cost=args.resale_fixed_cost,
        resale_safety_discount=args.resale_safety_discount,
        min_expected_profit=args.min_expected_profit,
        min_expected_profit_margin=args.min_expected_profit_margin,
    )

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
    if "DealEligible" in df_new.columns:
        print(f"- Eligible deals: {int(df_new['DealEligible'].sum())}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite index created by batch script")
    ap.add_argument("--out", required=True, help="CSV to write assigned new items to (can append)")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--new_items", help="CSV containing only new items from this run (recommended)")
    g.add_argument("--growing_csv", help="Growing CSV; script will process only appended rows using DB pointer")

    ap.add_argument("--append", action="store_true", help="Append to --out if exists")
    ap.add_argument("--price_buffer_size", type=int, default=200)
    ap.add_argument("--min_variant_size_for_confident_deals", type=int, default=8)
    ap.add_argument("--max_variant_mad_ratio", type=float, default=0.25)
    ap.add_argument("--hard_max_variant_mad_ratio", type=float, default=0.45)
    ap.add_argument("--min_deal_confidence", type=float, default=0.55)
    ap.add_argument("--min_centroid_similarity", type=float, default=0.55)
    ap.add_argument("--min_informative_tokens", type=int, default=2)
    ap.add_argument("--exclude_negflag_deals", action="store_true")
    ap.add_argument("--resale_fee_rate", type=float, default=0.10)
    ap.add_argument("--resale_fixed_cost", type=float, default=0.0)
    ap.add_argument("--resale_safety_discount", type=float, default=0.05)
    ap.add_argument("--min_expected_profit", type=float, default=0.0)
    ap.add_argument("--min_expected_profit_margin", type=float, default=0.0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
