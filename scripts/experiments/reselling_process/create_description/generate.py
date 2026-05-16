from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reselling_process.create_description.paths import (
    RUNS_DIR,
    SIMPLE_SCRAPE_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)


FULL_SCRAPE_SOURCES = (
    SIMPLE_SCRAPE_DIR / "full_scrape_merged_stage_resume_20260513" / "backfill_sold_unsold_all_searches.csv",
    SIMPLE_SCRAPE_DIR / "full_scrape_merged_stage_resume_20260513" / "main_full_scrape_verylow_veryhigh_all_searches.csv",
)
DEFAULT_OLLAMA_MODEL = "gemma2:2b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)
DEFAULT_GEMINI_TIMEOUT_SECONDS = 60
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 480
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

GENERIC_TITLE_TERMS = {
    "articolo",
    "item",
    "prodotto",
    "lotto",
    "scarpe",
    "borsa",
    "telefono",
    "accessorio",
}
STOPWORDS = {
    "a",
    "ad",
    "al",
    "alla",
    "con",
    "da",
    "dei",
    "del",
    "della",
    "di",
    "e",
    "ed",
    "for",
    "il",
    "in",
    "la",
    "le",
    "lo",
    "nuovo",
    "ottime",
    "per",
    "senza",
    "su",
    "the",
    "un",
    "una",
}
SELLER_NOISE_PATTERNS = (
    "guarda gli altri annunci",
    "guarda i miei annunci",
    "visita il mio profilo",
    "scrivimi in privato",
    "contattami",
    "non esitare a chiedere info",
    "prezzo trattabile",
    "preferisco la vendita tramite verifica vinted",
    "spedizione veloce",
    "tramite verifica vinted",
    "no scambi",
    "no perditempo",
    "bundle",
    "lotto disponibile",
)
INCLUDE_KEYWORDS = (
    "box",
    "scatola",
    "dust bag",
    "sacca",
    "scontrino",
    "ricevuta",
    "custodia",
    "caricatore",
    "cavo",
    "cartellino",
    "laccii",
    "lacci",
)
FLAW_KEYWORDS = (
    "difetto",
    "graff",
    "segn",
    "macchi",
    "usura",
    "rovinat",
    "manca",
    "scuc",
    "rott",
    "scolor",
    "piccolo difetto",
)
CONDITION_HINTS = (
    "nuovo con cartellino",
    "nuovo senza cartellino",
    "mai usato",
    "ottime condizioni",
    "buone condizioni",
    "condizioni perfette",
)
ITALIAN_SIGNAL_TERMS = {
    "accessori",
    "annuncio",
    "bellissimo",
    "buone",
    "colore",
    "completo",
    "condizioni",
    "cotone",
    "cristalli",
    "dettagli",
    "dustbag",
    "dust bag",
    "elegante",
    "facile",
    "indossata",
    "indossato",
    "mai",
    "modello",
    "nera",
    "nero",
    "ottime",
    "perfette",
    "perfetto",
    "presenta",
    "scatola",
    "segnalo",
    "segno",
    "taglia",
    "tessuto",
    "unisex",
    "usata",
    "usato",
    "vendo",
}
FOREIGN_MARKERS = {
    "brand new",
    "colour",
    "comes with",
    "excellent",
    "jamais",
    "neuf",
    "nuevo",
    "original design",
    "pink",
    "port\u00e9",
    "sous blister",
    "the worn",
    "with dust bag",
    "worn once",
}
SUPPORTED_LANGUAGES = {"english", "italian"}
LLM_BANNED_PATTERNS = (
    "this platform",
    "the listing",
    "seller text",
    "provided details",
    "signature style",
    "heritage",
    "investment piece",
    "collector",
    "boutique",
    "customer service",
    "shipping quality",
    "ready-to-wear",
    "directly from the seller",
)
LLM_CONDITION_UPGRADE_PATTERNS = (
    "pristine",
    "flawless",
    "untouched",
    "unworn",
    "perfect condition",
)
ENGLISH_FACT_REPLACEMENTS = (
    (r"\bnuovo con cartellino\b", "brand new with tags"),
    (r"\bnuovo senza cartellino\b", "brand new without tags"),
    (r"\bottime condizioni\b", "excellent condition"),
    (r"\bbuone condizioni\b", "good condition"),
    (r"\bneuf sous blister\b", "brand new, sealed in cellophane"),
    (r"\bmai portat[oa]\b", "never worn"),
    (r"\bmai usat[oa]\b", "never used"),
    (r"\bha la sua dustbag\b", "includes its dust bag"),
    (r"\bcon scatola originale\b", "includes the original box"),
    (r"\bcon la sua scatola\b", "includes its original box"),
    (r"\bdustbag\b", "dust bag"),
    (r"\bcartellino\b", "tags"),
    (r"\buomo\b", "men's"),
)


def normalize_id_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [token for token in text.split() if len(token) > 1 and token not in STOPWORDS]


def safe_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(price):
        return None
    return price


def parse_search_name(row: pd.Series | dict[str, Any]) -> str:
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)
    return clean_text(data.get("SearchName") or data.get("SourceSearch") or "")


def ensure_basic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def column_or_empty(name: str) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series("", index=out.index, dtype=object)

    if "SearchName" not in out.columns and "SourceSearch" in out.columns:
        out["SearchName"] = out["SourceSearch"]
    if "item_id" not in out.columns:
        out["item_id"] = out.get("Dataid", "").map(normalize_id_value) if "Dataid" in out.columns else ""
    link_values = column_or_empty("Link").map(clean_text)
    out["item_id"] = out["item_id"].where(out["item_id"].astype(str).str.len() > 0, link_values)
    if "Title_norm" not in out.columns:
        out["Title_norm"] = column_or_empty("Title").map(normalize_text)
    if "Brand_norm" not in out.columns:
        out["Brand_norm"] = column_or_empty("Brand").map(normalize_text)
    if "EmbedText" not in out.columns:
        out["EmbedText"] = (
            column_or_empty("Title").map(clean_text)
            + " | "
            + column_or_empty("Brand").map(clean_text)
            + " | "
            + column_or_empty("Size").map(clean_text)
        )
    if "Description" not in out.columns:
        out["Description"] = ""
    return out


def load_input_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return ensure_basic_columns(pd.read_csv(path, low_memory=False))


def choose_input_row(
    frame: pd.DataFrame,
    *,
    dataid: str | None,
    link: str | None,
    row_index: int | None,
) -> pd.Series:
    if dataid:
        item_id = normalize_id_value(dataid)
        matched = frame.loc[frame["item_id"].astype(str) == item_id]
        if matched.empty:
            raise ValueError(f"Dataid not found in input: {item_id}")
        return matched.iloc[0]
    if link:
        matched = frame.loc[frame.get("Link", pd.Series(index=frame.index, dtype=object)).astype(str) == str(link)]
        if matched.empty:
            raise ValueError("Link not found in input CSV")
        return matched.iloc[0]
    if row_index is None:
        if len(frame) != 1:
            raise ValueError("Use --dataid, --link, or --row-index when the input has multiple rows.")
        row_index = 0
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"row-index out of bounds: {row_index}")
    return frame.iloc[row_index]


def iter_local_item_sources(search_names: list[str] | None) -> list[Path]:
    if search_names:
        candidates = [SIMPLE_SCRAPE_DIR / search / "pipeline_out" / "items_with_product_and_variant.csv" for search in search_names]
    else:
        candidates = sorted(SIMPLE_SCRAPE_DIR.glob("*/pipeline_out/items_with_product_and_variant.csv"))
    return [path for path in candidates if path.exists()]


def read_item_source(path: Path) -> pd.DataFrame:
    keep = [
        "Title",
        "Price",
        "Brand",
        "Size",
        "Link",
        "Likes",
        "Dataid",
        "MarketStatus",
        "SearchName",
        "Title_norm",
        "Brand_norm",
        "BlockKey",
        "EmbedText",
        "ProductId",
        "VariantText",
        "VariantId",
        "ExpectedResalePrice",
        "ExpectedProfit",
        "ExpectedProfitMargin",
    ]
    df = pd.read_csv(path, low_memory=False)
    existing = [col for col in keep if col in df.columns]
    out = df.loc[:, existing].copy()
    out["SourceKind"] = "pipeline_out"
    out["SourcePath"] = str(path)
    return ensure_basic_columns(out)


def read_full_scrape_source(path: Path, search_names: list[str] | None) -> pd.DataFrame:
    keep = [
        "Title",
        "Price",
        "Brand",
        "Size",
        "Link",
        "Likes",
        "Dataid",
        "MarketStatus",
        "SearchName",
        "SourceSearch",
        "Description",
        "Condition",
        "SellerName",
        "SearchDate",
        "ObservedPagePrice",
    ]
    df = pd.read_csv(path, low_memory=False)
    existing = [col for col in keep if col in df.columns]
    out = df.loc[:, existing].copy()
    out = ensure_basic_columns(out)
    if search_names:
        out = out.loc[out["SearchName"].astype(str).isin(search_names)].copy()
    out["SourceKind"] = "full_scrape"
    out["SourcePath"] = str(path)
    return out.reset_index(drop=True)


def dedupe_corpus(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = ensure_basic_columns(frame)
    out["_description_len"] = out.get("Description", "").map(lambda x: len(clean_text(x)))
    out["_field_score"] = (
        out["_description_len"]
        + out.get("Condition", "").map(lambda x: len(clean_text(x)))
        + out.get("EmbedText", "").map(lambda x: len(clean_text(x)))
    )
    out = out.sort_values(by=["_field_score", "_description_len"], ascending=False)
    out = out.drop_duplicates(subset=["item_id"], keep="first")
    return out.drop(columns=["_description_len", "_field_score"]).reset_index(drop=True)


def load_example_corpus(
    *,
    search_names: list[str] | None,
    prefer_brand: str | None = None,
    all_searches: bool = False,
) -> pd.DataFrame:
    wanted = None if all_searches else search_names
    frames: list[pd.DataFrame] = []
    for path in iter_local_item_sources(wanted):
        frames.append(read_item_source(path))
    for path in FULL_SCRAPE_SOURCES:
        if path.exists():
            frames.append(read_full_scrape_source(path, wanted))
    if not frames:
        return ensure_basic_columns(pd.DataFrame())
    corpus = pd.concat(frames, ignore_index=True, sort=False)
    corpus = dedupe_corpus(corpus)
    if prefer_brand:
        brand_norm = normalize_text(prefer_brand)
        same_brand = corpus.loc[corpus["Brand_norm"].astype(str) == brand_norm].copy()
        if not same_brand.empty:
            return same_brand.reset_index(drop=True)
    return corpus.reset_index(drop=True)


def jaccard_score(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def similarity_score(query: pd.Series | dict[str, Any], candidate: pd.Series | dict[str, Any]) -> float:
    q = query if isinstance(query, dict) else query.to_dict()
    c = candidate if isinstance(candidate, dict) else candidate.to_dict()
    if normalize_id_value(q.get("item_id") or q.get("Dataid")) and normalize_id_value(q.get("item_id") or q.get("Dataid")) == normalize_id_value(c.get("item_id") or c.get("Dataid")):
        return float("-inf")

    score = 0.0
    q_brand = normalize_text(q.get("Brand_norm") or q.get("Brand"))
    c_brand = normalize_text(c.get("Brand_norm") or c.get("Brand"))
    if q_brand and c_brand:
        score += 3.0 if q_brand == c_brand else -1.5

    q_block = clean_text(q.get("BlockKey"))
    c_block = clean_text(c.get("BlockKey"))
    if q_block and c_block and q_block == c_block:
        score += 4.0

    q_product = normalize_id_value(q.get("ProductId"))
    c_product = normalize_id_value(c.get("ProductId"))
    if q_product and c_product and q_product == c_product and q_product != "0":
        score += 5.0

    q_variant = normalize_id_value(q.get("VariantId"))
    c_variant = normalize_id_value(c.get("VariantId"))
    if q_variant and c_variant and q_variant == c_variant and q_variant != "0":
        score += 6.0

    q_title_tokens = tokenize(q.get("Title"))
    c_title_tokens = tokenize(c.get("Title"))
    q_embed_tokens = tokenize(q.get("EmbedText"))
    c_embed_tokens = tokenize(c.get("EmbedText"))
    title_overlap = jaccard_score(q_title_tokens, c_title_tokens)
    embed_overlap = jaccard_score(q_embed_tokens, c_embed_tokens)
    score += 3.5 * title_overlap
    score += 2.5 * embed_overlap

    q_search = clean_text(q.get("SearchName"))
    c_search = clean_text(c.get("SearchName"))
    if q_search and c_search and q_search == c_search:
        score += 1.0

    q_price = safe_price(q.get("Price") or q.get("_PriceNum"))
    c_price = safe_price(c.get("Price") or c.get("ObservedPagePrice"))
    if q_price is not None and c_price is not None and q_price > 0:
        price_gap = abs(q_price - c_price) / max(q_price, 1.0)
        score += max(0.0, 1.5 - price_gap)

    if clean_text(c.get("Description")):
        score += 0.5
    is_sold = normalize_text(c.get("MarketStatus")) == "sold"
    if is_sold:
        score += 0.25
        if q_brand and c_brand and q_brand == c_brand and (title_overlap >= 0.45 or embed_overlap >= 0.5):
            score += 2.0
    return score


def retrieve_similar_examples(query_row: pd.Series, corpus: pd.DataFrame, *, limit: int = 5) -> pd.DataFrame:
    if corpus.empty:
        return corpus.copy()
    work = corpus.copy()
    work["similarity_score"] = work.apply(lambda row: similarity_score(query_row, row), axis=1)
    work = work.loc[work["similarity_score"] > 0].copy()
    if work.empty:
        return work
    market_status = work["MarketStatus"] if "MarketStatus" in work.columns else pd.Series("", index=work.index, dtype=object)
    descriptions = work["Description"] if "Description" in work.columns else pd.Series("", index=work.index, dtype=object)
    work["_is_sold"] = market_status.map(lambda value: normalize_text(value) == "sold")
    work["_description_len"] = descriptions.map(lambda value: len(clean_text(value)))
    work = work.sort_values(
        by=["similarity_score", "_is_sold", "_description_len", "SourceKind"],
        ascending=[False, False, False, True],
    )
    return work.drop(columns=["_is_sold", "_description_len"]).head(limit).reset_index(drop=True)


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"[\n\r]+|(?<=[\.!?;])\s+", cleaned)
    refined_parts: list[str] = []
    for part in parts:
        subparts = re.split(r"(?<=[a-z0-9\)])\s+(?=[A-Z][a-z])", part)
        for subpart in subparts:
            refined = subpart.strip(" -")
            if clean_text(refined):
                refined_parts.append(refined)
    return refined_parts


def is_noise_sentence(sentence: str) -> bool:
    lower = normalize_text(sentence)
    if len(lower) < 4:
        return True
    return any(pattern in lower for pattern in SELLER_NOISE_PATTERNS)


def clean_description_sentences(text: Any) -> list[str]:
    sentences = []
    for sentence in split_sentences(clean_text(text)):
        if is_noise_sentence(sentence):
            continue
        sentences.append(sentence)
    return sentences


def extract_matching_sentences(text: Any, keywords: tuple[str, ...]) -> list[str]:
    matches = []
    for sentence in clean_description_sentences(text):
        lower = normalize_text(sentence)
        if any(keyword in lower for keyword in keywords):
            matches.append(sentence)
    return matches


def infer_condition(row: pd.Series, similar_examples: pd.DataFrame) -> str:
    direct = clean_text(row.get("Condition"))
    if direct:
        return direct
    description = clean_text(row.get("Description"))
    desc_lower = normalize_text(description)
    for hint in CONDITION_HINTS:
        if hint in desc_lower:
            return hint.capitalize()
    if not similar_examples.empty and "Condition" in similar_examples.columns:
        common = [clean_text(value) for value in similar_examples["Condition"].dropna().astype(str) if clean_text(value)]
        if common:
            top, count = Counter(common).most_common(1)[0]
            if count >= 2:
                return top
    return ""


def derive_retrieval_keywords(query_row: pd.Series, similar_examples: pd.DataFrame, *, limit: int = 6) -> list[str]:
    if similar_examples.empty:
        return []
    query_tokens = set(tokenize(query_row.get("Title"))) | set(tokenize(query_row.get("Description")))
    counts: Counter[str] = Counter()
    for _, row in similar_examples.iterrows():
        counts.update(tokenize(row.get("Title")))
        counts.update(tokenize(row.get("Description")))
    keywords = []
    for token, freq in counts.most_common(20):
        if token in GENERIC_TITLE_TERMS or token in query_tokens or token in STOPWORDS:
            continue
        if freq < 2:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def sentence_key(text: str) -> str:
    return normalize_text(text)


def is_safe_borrowed_sentence(sentence: str) -> bool:
    lower = normalize_text(sentence)
    if not lower:
        return False
    if any(keyword in lower for keyword in INCLUDE_KEYWORDS):
        return False
    if any(keyword in lower for keyword in FLAW_KEYWORDS):
        return False
    if any(hint in lower for hint in CONDITION_HINTS):
        return False
    if re.search(r"\b\d{1,3}(?:[\.,]\d{1,2})?\b", lower):
        return False
    return True


def same_product_reference_match(query_row: pd.Series, candidate_row: pd.Series | dict[str, Any]) -> bool:
    candidate = candidate_row if isinstance(candidate_row, dict) else candidate_row.to_dict()
    q_brand = normalize_text(query_row.get("Brand_norm") or query_row.get("Brand"))
    c_brand = normalize_text(candidate.get("Brand_norm") or candidate.get("Brand"))
    if not q_brand or not c_brand or q_brand != c_brand:
        return False

    query_title_tokens = tokenize(query_row.get("Title"))
    candidate_title_tokens = tokenize(candidate.get("Title"))
    title_overlap = jaccard_score(query_title_tokens, candidate_title_tokens)
    if title_overlap >= 0.45:
        return True

    similarity = candidate.get("similarity_score")
    try:
        return float(similarity) >= 7.0
    except (TypeError, ValueError):
        return False


def collect_sold_reference_sentences(query_row: pd.Series, similar_examples: pd.DataFrame, *, limit: int = 2) -> list[str]:
    if similar_examples.empty:
        return []

    original_keys = {sentence_key(sentence) for sentence in clean_description_sentences(query_row.get("Description"))}
    collected: list[str] = []
    seen_keys = set(original_keys)
    for _, row in similar_examples.iterrows():
        if normalize_text(row.get("MarketStatus")) != "sold":
            continue
        if not same_product_reference_match(query_row, row):
            continue
        for sentence in clean_description_sentences(row.get("Description")):
            key = sentence_key(sentence)
            if key in seen_keys or not is_safe_borrowed_sentence(sentence):
                continue
            collected.append(sentence)
            seen_keys.add(key)
            if len(collected) >= limit:
                return collected
    return collected


def normalize_size_fact(value: Any) -> str:
    cleaned = clean_text(value)
    normalized = normalize_text(cleaned)
    if normalized in {"", "0", "0.0", "00", "none", "nan", "n/a", "na", "null"}:
        return ""
    return cleaned


def build_structured_facts(query_row: pd.Series, similar_examples: pd.DataFrame) -> dict[str, Any]:
    kept_sentences = clean_description_sentences(query_row.get("Description"))
    accessory_sentences = extract_matching_sentences(query_row.get("Description"), INCLUDE_KEYWORDS)
    flaw_sentences = extract_matching_sentences(query_row.get("Description"), FLAW_KEYWORDS)
    condition = infer_condition(query_row, similar_examples)
    sold_reference_sentences = collect_sold_reference_sentences(query_row, similar_examples)
    facts = {
        "title": clean_text(query_row.get("Title")),
        "brand": clean_text(query_row.get("Brand")),
        "size": normalize_size_fact(query_row.get("Size")),
        "search_name": clean_text(query_row.get("SearchName")),
        "condition": condition,
        "price": safe_price(query_row.get("Price") or query_row.get("_PriceNum")),
        "original_description": clean_text(query_row.get("Description")),
        "factual_sentences": kept_sentences[:3],
        "sold_reference_sentences": sold_reference_sentences,
        "accessory_notes": accessory_sentences[:2],
        "flaw_notes": flaw_sentences[:2],
        "retrieval_keywords": derive_retrieval_keywords(query_row, similar_examples),
    }
    return facts


def title_is_generic(title: str) -> bool:
    tokens = tokenize(title)
    if not tokens:
        return True
    return all(token in GENERIC_TITLE_TERMS for token in tokens)


def choose_heading(facts: dict[str, Any], similar_examples: pd.DataFrame) -> str:
    title = facts.get("title") or ""
    brand = facts.get("brand") or ""
    if title and not title_is_generic(title):
        if brand and normalize_text(brand) not in normalize_text(title):
            return f"{brand} {title}".strip()
        return title
    if not similar_examples.empty:
        top_title = clean_text(similar_examples.iloc[0].get("Title"))
        if top_title:
            return top_title
    return brand or title or "Articolo usato"


def ensure_sentence(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def lower_first(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return cleaned[:1].lower() + cleaned[1:]


def contains_condition_hint(text: str) -> bool:
    lower = normalize_text(text)
    return any(hint in lower for hint in CONDITION_HINTS)


def looks_like_italian_listing_sentence(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False

    lower = normalize_text(cleaned)
    foreign_hits = sum(1 for marker in FOREIGN_MARKERS if marker in lower)
    if foreign_hits >= 1:
        return False

    tokens = lower.split()
    if not tokens:
        return False

    stopword_hits = sum(1 for token in tokens if token in STOPWORDS)
    signal_hits = sum(1 for token in tokens if token in ITALIAN_SIGNAL_TERMS)
    starts_like_listing = lower.startswith((
        "con ",
        "senza ",
        "ottime",
        "buone",
        "mai ",
        "indossat",
        "modello",
        "articolo",
        "vendo",
        "perfett",
    ))
    return starts_like_listing or signal_hits >= 2 or (signal_hits >= 1 and stopword_hits >= 1) or stopword_hits >= 2


def pick_narrative_detail_sentences(sentences: list[str], *, limit: int = 2) -> list[str]:
    picked: list[str] = []
    for sentence in sentences:
        if contains_condition_hint(sentence):
            continue
        if extract_matching_sentences(sentence, INCLUDE_KEYWORDS):
            continue
        if extract_matching_sentences(sentence, FLAW_KEYWORDS):
            continue
        if not looks_like_italian_listing_sentence(sentence):
            continue
        picked.append(ensure_sentence(sentence))
        if len(picked) >= limit:
            break
    return picked


def rewrite_accessory_sentence(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    lower = normalize_text(cleaned)
    if lower.startswith("con "):
        return ensure_sentence("Completo di " + cleaned[4:])
    if lower.startswith("ha ") or lower.startswith("dotato "):
        return ensure_sentence(cleaned)
    return ensure_sentence(cleaned)


def rewrite_flaw_sentence(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    lowered = lower_first(cleaned)
    if lowered.startswith(("un ", "una ", "piccol", "liev", "qualche ")):
        return ensure_sentence(f"Segnalo solo {lowered}")
    return ensure_sentence(f"Segnalo solo {lowered}")


def generate_description_text(facts: dict[str, Any], similar_examples: pd.DataFrame) -> str:
    heading = choose_heading(facts, similar_examples)
    intro_bits = []
    if heading:
        intro_bits.append(heading)
    if facts.get("size"):
        intro_bits.append(f"taglia/variante {facts['size']}")
    if facts.get("condition"):
        intro_bits.append(lower_first(facts["condition"]))

    paragraphs: list[str] = []
    if intro_bits:
        intro = "Vendo " + ", ".join(intro_bits)
        paragraphs.append(ensure_sentence(intro))

    body_sentences: list[str] = []
    factual_sentences = facts.get("factual_sentences") or []
    for sentence in pick_narrative_detail_sentences(factual_sentences, limit=2):
        normalized_sentence = normalize_text(sentence)
        if normalized_sentence and all(normalized_sentence != normalize_text(existing) for existing in body_sentences):
            body_sentences.append(sentence)

    for sentence in (facts.get("sold_reference_sentences") or [])[:2]:
        if not looks_like_italian_listing_sentence(sentence):
            continue
        normalized_sentence = normalize_text(sentence)
        if normalized_sentence and all(normalized_sentence != normalize_text(existing) for existing in body_sentences):
            body_sentences.append(ensure_sentence(sentence))

    accessory_notes = facts.get("accessory_notes") or []
    if accessory_notes:
        if not looks_like_italian_listing_sentence(accessory_notes[0]):
            accessory_notes = []
    if accessory_notes:
        accessory_text = rewrite_accessory_sentence(accessory_notes[0])
        normalized_accessory = normalize_text(accessory_text)
        if all(normalized_accessory != normalize_text(existing) for existing in body_sentences):
            body_sentences.append(accessory_text)

    flaw_notes = facts.get("flaw_notes") or []
    if flaw_notes:
        flaw_text = rewrite_flaw_sentence(flaw_notes[0])
        normalized_flaw = normalize_text(flaw_text)
        if normalized_flaw and all(normalized_flaw != normalize_text(existing) for existing in body_sentences):
            body_sentences.append(flaw_text)

    if body_sentences:
        paragraphs.append(" ".join(body_sentences))

    paragraphs.append("Se vuoi altre foto o misure scrivimi pure.")
    return "\n\n".join(paragraph.strip() for paragraph in paragraphs if clean_text(paragraph)).strip()


def concise_price_text(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def translate_text_for_prompt(text: Any, *, language: str) -> str:
    cleaned = clean_text(text)
    if not cleaned or language != "english":
        return cleaned

    translated = cleaned
    for pattern, replacement in ENGLISH_FACT_REPLACEMENTS:
        translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


def build_prompt_characteristics_block(
    facts: dict[str, Any],
    similar_examples: pd.DataFrame,
    *,
    language: str = "english",
) -> str:
    lines = [
        f"- Item: {translate_text_for_prompt(facts.get('title') or 'Unknown', language=language) or 'Unknown'}",
        f"- Brand: {translate_text_for_prompt(facts.get('brand') or 'Unknown', language=language) or 'Unknown'}",
    ]
    if facts.get("size"):
        lines.append(f"- Size or variant: {translate_text_for_prompt(facts['size'], language=language)}")
    if facts.get("condition"):
        lines.append(f"- Condition: {translate_text_for_prompt(facts['condition'], language=language)}")
    if facts.get("price") is not None:
        lines.append(f"- Price: EUR {concise_price_text(facts['price'])}")

    detail_sentences = [translate_text_for_prompt(sentence, language=language) for sentence in (facts.get("factual_sentences") or []) if clean_text(sentence)]
    if detail_sentences:
        lines.append(f"- Seller-provided details: {' | '.join(detail_sentences[:3])}")

    accessory_notes = [translate_text_for_prompt(sentence, language=language) for sentence in (facts.get("accessory_notes") or []) if clean_text(sentence)]
    if accessory_notes:
        lines.append(f"- Accessories or extras: {' | '.join(accessory_notes[:2])}")

    flaw_notes = [translate_text_for_prompt(sentence, language=language) for sentence in (facts.get("flaw_notes") or []) if clean_text(sentence)]
    if flaw_notes:
        lines.append(f"- Flaws to mention carefully: {' | '.join(flaw_notes[:2])}")

    retrieval_cues: list[str] = []
    for _, row in similar_examples.head(3).iterrows():
        title = translate_text_for_prompt(row.get("Title"), language=language)
        description = translate_text_for_prompt(row.get("Description"), language=language)
        if title and description:
            retrieval_cues.append(f"{title}: {description[:160]}")
        elif title:
            retrieval_cues.append(title)
    if retrieval_cues:
        lines.append(f"- Similar listing cues: {' || '.join(retrieval_cues)}")

    return "\n".join(lines)


def build_llama_description_prompt(
    facts: dict[str, Any],
    similar_examples: pd.DataFrame,
    *,
    language: str,
) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    language_label = "English" if language == "english" else "Italian"
    language_hint = (
        "Write in polished marketplace English."
        if language == "english"
        else "Write in polished marketplace Italian."
    )
    closing_hint = (
        "End with a light invitation to message for more photos or details."
        if language == "english"
        else "Chiudi con un invito leggero a scrivere per altre foto o dettagli."
    )

    return (
        f"You are writing a second-hand marketplace description in {language_label}.\n"
        f"{language_hint}\n"
        "The final output must be fully in English from start to finish. Translate non-English source details into natural English, or omit them if they do not translate cleanly.\n"
        "Make it very appealing, human, and credible, like an experienced reseller who knows how to present an item well without sounding fake.\n"
        "Use only the facts provided below. Do not invent authenticity claims, measurements, accessories, defects, usage history, packaging, or seller guarantees.\n"
        "If a detail is not provided, omit it completely instead of guessing.\n"
        "Keep the condition wording close to the provided facts. Do not upgrade it into stronger claims like pristine, flawless, untouched, unworn, or perfect unless those words are explicitly supported by the facts.\n"
        "You may add light persuasive language about the item's style, scent, elegance, wearability, or overall appeal, as long as you do not turn that into invented concrete facts.\n"
        "Do not add generic luxury filler such as investment piece, rare, collector, premium craftsmanship, or heritage language unless those ideas are directly supported by the facts.\n"
        "Do not mention boutiques, stores, shipping quality, customer service, verification, or invitations to negotiate unless they are explicitly present in the facts.\n"
        "Do not refer to the listing, the seller text, or the provided details as sources. State the facts naturally as part of the description.\n"
        "Do not mention that you are using notes or extracted characteristics. Do not use bullet points.\n"
        "Write one short paragraph of 3 to 4 sentences.\n"
        "Sentence 1 should identify the item, brand, and condition in a natural way.\n"
        "Use the next sentences for only the strongest factual details, accessories, or one careful flaw mention if provided.\n"
        "Keep the tone elegant and persuasive, but specific and restrained.\n"
        f"{closing_hint}\n\n"
        "Facts about the item:\n"
        f"{build_prompt_characteristics_block(facts, similar_examples, language=language)}\n"
    )


def ollama_model_installed(model: str) -> bool:
    result = subprocess.run(
        ["ollama", "list"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return False
    return any(line.split() and line.split()[0] == model for line in result.stdout.splitlines()[1:])


def ensure_ollama_model(model: str, *, pull_if_missing: bool) -> None:
    if ollama_model_installed(model):
        return
    if not pull_if_missing:
        raise RuntimeError(
            f"Ollama model '{model}' is not installed. Run 'ollama pull {model}' or pass --ollama-pull-model."
        )
    pull = subprocess.run(["ollama", "pull", model], check=False, text=True)
    if pull.returncode != 0:
        raise RuntimeError(f"Failed to pull Ollama model '{model}'.")


def format_ollama_runtime_error(message: str, *, model: str) -> str:
    memory_match = re.search(
        r"model requires more system memory \(([^)]+)\) than is available \(([^)]+)\)",
        message,
        flags=re.IGNORECASE,
    )
    if memory_match:
        required_memory, available_memory = memory_match.groups()
        return (
            f"Ollama cannot run '{model}' on this machine right now: it requires {required_memory} of system memory, "
            f"but only {available_memory} is currently available. Free RAM and swap, close other memory-heavy processes, "
            f"or use a smaller local model."
        )
    return message or f"Ollama run failed for model '{model}'."


def llm_output_needs_fallback(text: str, facts: dict[str, Any]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    if any(pattern in normalized for pattern in LLM_BANNED_PATTERNS):
        return True

    condition = normalize_text(facts.get("condition"))
    if condition and any(pattern in normalized for pattern in LLM_CONDITION_UPGRADE_PATTERNS):
        return True
    if "nuovo" in condition and any(pattern in normalized for pattern in ("pre-owned", "used condition", "worn")):
        return True
    if "senza cartellino" in condition and "price tag" in normalized:
        return True
    return False


def generate_description_with_ollama(
    facts: dict[str, Any],
    similar_examples: pd.DataFrame,
    *,
    model: str,
    language: str,
    pull_if_missing: bool,
) -> tuple[str, str]:
    ensure_ollama_model(model, pull_if_missing=pull_if_missing)
    prompt = build_llama_description_prompt(facts, similar_examples, language=language)
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(format_ollama_runtime_error(result.stderr.strip(), model=model))
    output_text = clean_text(result.stdout)
    if llm_output_needs_fallback(output_text, facts):
        return generate_description_text(facts, similar_examples), prompt
    return output_text, prompt


def extract_gemini_response_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates") or []
    text_parts: list[str] = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = clean_text(part.get("text"))
            if text:
                text_parts.append(text)
    return "\n".join(text_parts).strip()


def extract_gemini_finish_reason(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates") or []
    for candidate in candidates:
        finish_reason = clean_text(candidate.get("finishReason"))
        if finish_reason:
            return finish_reason
    return ""


def format_gemini_runtime_error(message: str, *, model: str) -> str:
    cleaned_message = clean_text(message)
    if not cleaned_message:
        return f"Gemini API request for '{model}' failed."
    return f"Gemini API request for '{model}' failed: {cleaned_message}"


def is_gemini_quota_error(message: str) -> bool:
    normalized = normalize_text(message)
    return "quota exceeded" in normalized or "resource exhausted" in normalized


def build_gemini_model_candidates(preferred_model: str) -> list[str]:
    normalized_preferred = clean_text(preferred_model) or DEFAULT_GEMINI_MODEL
    candidates = [normalized_preferred]
    for model in GEMINI_FALLBACK_MODELS:
        if model not in candidates:
            candidates.append(model)
    return candidates


def build_gemini_generation_config(model: str, *, max_output_tokens: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "temperature": 0.85,
        "topP": 0.95,
        "maxOutputTokens": max_output_tokens,
    }
    normalized_model = clean_text(model)
    if normalized_model.startswith("gemini-2.5-flash"):
        config["thinkingConfig"] = {"thinkingBudget": 0}
    return config


def gemini_output_looks_incomplete(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return True
    if cleaned[-1] not in ".!?":
        return True
    return len(split_sentences(cleaned)) < 2


def generate_description_with_gemini_api(
    facts: dict[str, Any],
    similar_examples: pd.DataFrame,
    *,
    model: str,
    language: str,
    api_key: str | None,
    timeout_seconds: int = DEFAULT_GEMINI_TIMEOUT_SECONDS,
    max_output_tokens: int = DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    prompt_override: str | None = None,
    allow_retry: bool = True,
) -> tuple[str, str]:
    resolved_api_key = clean_text(api_key or os.getenv("GEMINI_API_KEY"))
    if not resolved_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env to enable Gemini description generation.")

    prompt = prompt_override or build_llama_description_prompt(facts, similar_examples, language=language)
    response = requests.post(
        GEMINI_API_URL_TEMPLATE.format(model=model),
        params={"key": resolved_api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": build_gemini_generation_config(model, max_output_tokens=max_output_tokens),
        },
        timeout=timeout_seconds,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        error_payload: Any
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = response.text
        raise RuntimeError(format_gemini_runtime_error(str(error_payload), model=model)) from exc

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise RuntimeError(format_gemini_runtime_error("invalid JSON response", model=model)) from exc

    output_text = clean_text(extract_gemini_response_text(response_payload))
    finish_reason = extract_gemini_finish_reason(response_payload)
    if finish_reason == "MAX_TOKENS" and max_output_tokens < 768:
        return generate_description_with_gemini_api(
            facts,
            similar_examples,
            model=model,
            language=language,
            api_key=resolved_api_key,
            timeout_seconds=timeout_seconds,
            max_output_tokens=768,
            prompt_override=prompt,
            allow_retry=False,
        )
    if not output_text:
        raise RuntimeError(format_gemini_runtime_error("empty response body", model=model))
    if allow_retry and (gemini_output_looks_incomplete(output_text) or llm_output_needs_fallback(output_text, facts)):
        retry_prompt = (
            prompt
            + "\n\nReturn one complete paragraph of 3 to 4 full sentences."
            + " End with a full sentence and final punctuation."
            + " Do not stop mid-sentence."
        )
        retry_text, retry_prompt_used = generate_description_with_gemini_api(
            facts,
            similar_examples,
            model=model,
            language=language,
            api_key=resolved_api_key,
            timeout_seconds=timeout_seconds,
            max_output_tokens=768,
            prompt_override=retry_prompt,
            allow_retry=False,
        )
        if not gemini_output_looks_incomplete(retry_text):
            return retry_text, retry_prompt_used
    return output_text, prompt


def build_description_payload(
    query_row: pd.Series,
    *,
    corpus: pd.DataFrame,
    top_k: int = 5,
    use_ollama: bool = False,
    use_gemini_api: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_language: str = "english",
    ollama_pull_model: bool = False,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_api_key: str | None = None,
) -> dict[str, Any]:
    similar_examples = retrieve_similar_examples(query_row, corpus, limit=top_k)
    facts = build_structured_facts(query_row, similar_examples)
    llm_prompt = ""
    generation_mode = "heuristic"
    model_name = ""
    if use_gemini_api:
        last_gemini_error: RuntimeError | None = None
        for candidate_model in build_gemini_model_candidates(gemini_model):
            try:
                description_text, llm_prompt = generate_description_with_gemini_api(
                    facts,
                    similar_examples,
                    model=candidate_model,
                    language=ollama_language,
                    api_key=gemini_api_key,
                )
                generation_mode = "gemini_api"
                model_name = candidate_model
                break
            except RuntimeError as exc:
                last_gemini_error = exc
                if not is_gemini_quota_error(str(exc)):
                    raise
        else:
            assert last_gemini_error is not None
            raise last_gemini_error
    elif use_ollama:
        description_text, llm_prompt = generate_description_with_ollama(
            facts,
            similar_examples,
            model=ollama_model,
            language=ollama_language,
            pull_if_missing=ollama_pull_model,
        )
        generation_mode = "ollama"
        model_name = ollama_model
    else:
        description_text = generate_description_text(facts, similar_examples)
    return {
        "query_item": {key: query_row.get(key) for key in query_row.index},
        "structured_facts": facts,
        "generated_description": description_text,
        "retrieved_examples": similar_examples.to_dict(orient="records"),
        "generation_mode": generation_mode,
        "model_name": model_name,
        "llm_prompt": llm_prompt,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a retrieval-grounded resale description for one item from the local project corpus."
    )
    ap.add_argument("--input", required=True, help="Path to the source CSV containing the item to describe.")
    ap.add_argument("--dataid", default="", help="Select the item by Dataid.")
    ap.add_argument("--link", default="", help="Select the item by full listing URL.")
    ap.add_argument("--row-index", type=int, default=None, help="Select the item by row index when input has multiple rows.")
    ap.add_argument("--top-k", type=int, default=5, help="Number of retrieved examples to keep.")
    ap.add_argument("--all-searches", action="store_true", help="Load the corpus from every search instead of restricting to the input search.")
    ap.add_argument("--out-dir", default=None, help="Optional output directory under the experiment root.")
    ap.add_argument("--use-ollama", action="store_true", help="Use a local Ollama model to write the final description.")
    ap.add_argument("--use-gemini-api", action="store_true", help="Use Gemini API to write the final description.")
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model to use for the final description.")
    ap.add_argument("--gemini-api-key", default="", help="Optional Gemini API key. Falls back to GEMINI_API_KEY from the environment.")
    ap.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model to use for the final description.")
    ap.add_argument("--ollama-language", default="english", choices=sorted(SUPPORTED_LANGUAGES), help="Output language for the Ollama-generated description.")
    ap.add_argument("--ollama-pull-model", action="store_true", help="Pull the Ollama model automatically if it is missing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_experiment_dirs()

    input_path = Path(args.input)
    frame = load_input_frame(input_path)
    query_row = choose_input_row(frame, dataid=args.dataid or None, link=args.link or None, row_index=args.row_index)
    search_name = parse_search_name(query_row)
    corpus = load_example_corpus(
        search_names=[search_name] if search_name else None,
        prefer_brand=clean_text(query_row.get("Brand")),
        all_searches=args.all_searches,
    )
    try:
        payload = build_description_payload(
            query_row,
            corpus=corpus,
            top_k=args.top_k,
            use_ollama=args.use_ollama,
            use_gemini_api=args.use_gemini_api,
            ollama_model=args.ollama_model,
            ollama_language=args.ollama_language,
            ollama_pull_model=args.ollama_pull_model,
            gemini_model=args.gemini_model,
            gemini_api_key=args.gemini_api_key or None,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None

    out_dir = Path(args.out_dir) if args.out_dir else RUNS_DIR / run_id("description")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "description_payload.json", payload)
    (out_dir / "generated_description.txt").write_text(payload["generated_description"] + "\n", encoding="utf-8")
    pd.DataFrame(payload["retrieved_examples"]).to_csv(out_dir / "retrieved_examples.csv", index=False)
    write_manifest(
        out_dir / "manifest.json",
        command="reselling_process.create_description.generate",
        extra={
            "input": str(input_path),
            "selected_item_id": normalize_id_value(query_row.get("item_id") or query_row.get("Dataid")),
            "search_name": search_name,
            "top_k": args.top_k,
            "corpus_rows": int(len(corpus)),
            "retrieved_rows": int(len(payload["retrieved_examples"])),
            "generation_mode": payload.get("generation_mode", "heuristic"),
            "model_name": payload.get("model_name", ""),
        },
    )
    print(payload["generated_description"])
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()