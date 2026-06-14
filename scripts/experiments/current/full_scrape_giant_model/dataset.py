from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.full_scrape_model.dataset import TASK_SOLD_STATUS, build_search_dataset
from experiments.current.full_scrape_giant_model.full_scrape_features import add_full_scrape_features
from experiments.deal_finder.modeling import ELIGIBLE_COL, TARGET_COL
from experiments.deal_finder import model_sweep as base_sweep


MAIN_SEARCHES = ("nike", "griffati_donna_all", "griffati_uomo_all", "gucci", "prada", "ps4", "Borse_Griffate")


def load_giant_frame(searches=MAIN_SEARCHES) -> pd.DataFrame:
    """Load and concatenate full-scrape datasets for requested searches."""
    frames = []
    for search in searches:
        frame = build_search_dataset(TASK_SOLD_STATUS, search)
        if frame.empty:
            print(f"[full_scrape_giant] WARN: no rows for {search}", flush=True)
            continue
        frames.append(frame)
    if not frames:
        raise ValueError("no rows for any requested search")
    combined = pd.concat(frames, ignore_index=True)
    combined["SearchName"] = combined["SearchName"].fillna("").astype(str)
    return combined


def prepare_full_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare full-scrape frame: filter, dedupe, engineer features."""
    out = df.copy()
    if ELIGIBLE_COL not in out.columns or TARGET_COL not in out.columns:
        return out.iloc[0:0].copy()
    eligible = base_sweep.parse_bool_series(out[ELIGIBLE_COL])
    out = out[eligible].copy()
    out["item_id"] = out.get("item_id", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    out = out[out["item_id"].str.len() > 0].copy()
    out = out.drop_duplicates(subset=["SearchName", "item_id"], keep="last")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce").fillna(0).astype(int)
    # Engineered snapshot numerics (price_num, likes_num, page_num, ratios) are
    # required by the rules baseline and the engineered-logistic spec; full-scrape
    # features add the item/seller metadata + Title/Description text columns.
    out = base_sweep.add_engineered_snapshot_features(out)
    out = add_full_scrape_features(out)
    return out.reset_index(drop=True)


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dataset statistics by search."""
    return (
        frame.groupby("SearchName", as_index=False)
        .agg(
            rows=("SearchName", "size"),
            positive_rows=(TARGET_COL, "sum"),
        )
        .assign(base_rate=lambda x: x["positive_rows"] / x["rows"])
    )
