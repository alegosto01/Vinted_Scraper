import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep


SOLD_STATUS_RUN = ROOT / "data" / "experiments" / "full_scrape_model" / "offline_runs" / "sold_status_sweep_20260515_101018"
BEST_BY_SEARCH_PATH = SOLD_STATUS_RUN / "best_by_search.csv"
DATASETS_DIR = SOLD_STATUS_RUN / "datasets"
EXCLUDED_SEARCHES = {"Borse_Griffate", "Scarpe_Griffate"}
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
SEEDS = (42, 123, 2026)


@dataclass(frozen=True)
class SearchCurveSummary:
    search_name: str
    approach: str
    current_pr_auc: float | None
    current_precision: float | None
    current_p10: float | None
    pr_auc_gain_80_to_100: float | None
    precision_gain_80_to_100: float | None
    p10_gain_80_to_100: float | None
    pr_auc_gain_20_to_100: float | None
    recommendation: str
    rationale: str


def best_specs() -> dict[str, model_sweep.ApproachSpec]:
    spec_by_name = {spec.name: spec for spec in model_sweep.APPROACHES}
    best = pd.read_csv(BEST_BY_SEARCH_PATH)
    out: dict[str, model_sweep.ApproachSpec] = {}
    for row in best.to_dict(orient="records"):
        search_name = str(row.get("search_name", ""))
        if search_name in EXCLUDED_SEARCHES:
            continue
        approach = str(row.get("approach", ""))
        if approach in spec_by_name:
            out[search_name] = spec_by_name[approach]
    return out


def stratified_fraction_sample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 0.999:
        return frame.reset_index(drop=True)
    if frame.empty:
        return frame.copy()
    rng = np.random.default_rng(seed)
    parts = []
    for label, part in frame.groupby(model_sweep.TARGET_COL, sort=False):
        n = max(1, int(np.floor(len(part) * fraction)))
        n = min(len(part), n)
        parts.append(part.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_search_frame(search_name: str, spec: model_sweep.ApproachSpec) -> pd.DataFrame:
    df = pd.read_csv(DATASETS_DIR / f"{search_name}.csv", low_memory=False)
    frame = model_sweep.prepare_sweep_frame(df)
    frame = model_sweep.add_engineered_snapshot_features(frame)
    if spec.use_image:
        frame = model_sweep.add_image_features(frame)
    return frame


def evaluate_fraction(frame: pd.DataFrame, spec: model_sweep.ApproachSpec, fraction: float, seed: int) -> dict[str, float | int | None]:
    splits = model_sweep.stratified_random_split(frame, seed=seed)
    if min(len(splits.train), len(splits.validation), len(splits.test)) == 0:
        return {"status": "skipped"}
    train_subset = stratified_fraction_sample(splits.train, fraction, seed)
    if train_subset[model_sweep.TARGET_COL].nunique() < 2:
        return {"status": "skipped"}
    numeric, text = model_sweep.select_features(train_subset, spec)
    if not numeric and not text:
        return {"status": "skipped"}
    if spec.kind == "linear_svm_calibrated" and train_subset[model_sweep.TARGET_COL].value_counts().min() < 3:
        return {"status": "skipped"}
    fit_frame = model_sweep.bounded_fit_frame(train_subset, spec, seed)
    model = model_sweep.make_model(spec, numeric, text)
    model.fit(fit_frame, fit_frame[model_sweep.TARGET_COL].astype(int))
    val_scores = model_sweep.score_with_model(model, splits.validation)
    test_scores = model_sweep.score_with_model(model, splits.test)
    threshold = model_sweep.choose_threshold(
        splits.validation[model_sweep.TARGET_COL].astype(int).to_numpy(),
        val_scores,
        min_precision=model_sweep.PROMOTION_PRECISION,
        min_count=model_sweep.VALIDATION_MIN_COUNT,
    )["threshold"]
    metrics = model_sweep.evaluate_scores(splits.test, test_scores, float(threshold))
    return {
        "status": "ok",
        "rows": int(len(train_subset)),
        "test_precision": metrics.get("threshold", {}).get("precision"),
        "test_count": metrics.get("threshold", {}).get("count"),
        "test_precision_at_10": metrics.get("precision_at", {}).get("p@10", {}).get("precision"),
        "test_pr_auc": metrics.get("pr_auc"),
        "test_roc_auc": metrics.get("roc_auc"),
    }


def aggregate_learning_curve(search_name: str, spec: model_sweep.ApproachSpec) -> pd.DataFrame:
    frame = prepare_search_frame(search_name, spec)
    rows: list[dict[str, object]] = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            metrics = evaluate_fraction(frame, spec, fraction, seed)
            rows.append(
                {
                    "search_name": search_name,
                    "approach": spec.name,
                    "fraction": fraction,
                    "seed": seed,
                    **metrics,
                }
            )
    raw = pd.DataFrame(rows)
    usable = raw[raw["status"] == "ok"].copy()
    grouped = (
        usable.groupby(["search_name", "approach", "fraction"], as_index=False)
        .agg(
            runs=("seed", "count"),
            train_rows_mean=("rows", "mean"),
            test_precision_mean=("test_precision", "mean"),
            test_precision_at_10_mean=("test_precision_at_10", "mean"),
            test_pr_auc_mean=("test_pr_auc", "mean"),
            test_roc_auc_mean=("test_roc_auc", "mean"),
            test_count_mean=("test_count", "mean"),
        )
        .sort_values("fraction")
        .reset_index(drop=True)
    )
    return grouped


def _safe_delta(table: pd.DataFrame, col: str, start_fraction: float, end_fraction: float) -> float | None:
    start = table.loc[np.isclose(table["fraction"], start_fraction), col]
    end = table.loc[np.isclose(table["fraction"], end_fraction), col]
    if start.empty or end.empty or pd.isna(start.iloc[0]) or pd.isna(end.iloc[0]):
        return None
    return float(end.iloc[0] - start.iloc[0])


def recommendation_from_curve(table: pd.DataFrame) -> SearchCurveSummary:
    search_name = str(table.iloc[0]["search_name"])
    approach = str(table.iloc[0]["approach"])
    current = table.loc[np.isclose(table["fraction"], 1.0)].iloc[0]
    gain_80_to_100_auc = _safe_delta(table, "test_pr_auc_mean", 0.8, 1.0)
    gain_80_to_100_precision = _safe_delta(table, "test_precision_mean", 0.8, 1.0)
    gain_80_to_100_p10 = _safe_delta(table, "test_precision_at_10_mean", 0.8, 1.0)
    gain_20_to_100_auc = _safe_delta(table, "test_pr_auc_mean", 0.2, 1.0)

    current_auc = current.get("test_pr_auc_mean")
    current_precision = current.get("test_precision_mean")
    current_p10 = current.get("test_precision_at_10_mean")

    if (
        gain_80_to_100_auc is not None
        and gain_80_to_100_precision is not None
        and gain_80_to_100_auc >= 0.015
        and gain_80_to_100_precision >= 0.02
    ):
        recommendation = "keep_scraping"
        rationale = "performance is still improving noticeably from 80% to 100% of current data"
    elif (
        gain_80_to_100_auc is not None
        and gain_80_to_100_precision is not None
        and gain_80_to_100_auc <= 0.005
        and gain_80_to_100_precision <= 0.01
        and current_auc is not None
        and current_precision is not None
        and float(current_auc) >= 0.75
        and float(current_precision) >= 0.85
    ):
        recommendation = "near_cap"
        rationale = "the curve is flattening and the current model is already strong"
    elif (
        gain_80_to_100_auc is not None
        and gain_80_to_100_auc <= 0.005
        and current_auc is not None
        and float(current_auc) < 0.7
    ):
        recommendation = "model_limited"
        rationale = "more of the same data is not helping much; model/features likely matter more now"
    else:
        recommendation = "mixed"
        rationale = "there may be some benefit left, but the gain is not strong enough to call it clearly"

    return SearchCurveSummary(
        search_name=search_name,
        approach=approach,
        current_pr_auc=None if pd.isna(current_auc) else float(current_auc),
        current_precision=None if pd.isna(current_precision) else float(current_precision),
        current_p10=None if pd.isna(current_p10) else float(current_p10),
        pr_auc_gain_80_to_100=None if gain_80_to_100_auc is None else float(gain_80_to_100_auc),
        precision_gain_80_to_100=None if gain_80_to_100_precision is None else float(gain_80_to_100_precision),
        p10_gain_80_to_100=None if gain_80_to_100_p10 is None else float(gain_80_to_100_p10),
        pr_auc_gain_20_to_100=None if gain_20_to_100_auc is None else float(gain_20_to_100_auc),
        recommendation=recommendation,
        rationale=rationale,
    )


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_tables: list[pd.DataFrame] = []
    summaries: list[SearchCurveSummary] = []
    for search_name, spec in best_specs().items():
        curve = aggregate_learning_curve(search_name, spec)
        if curve.empty:
            continue
        curve_tables.append(curve)
        summaries.append(recommendation_from_curve(curve))
    curves = pd.concat(curve_tables, ignore_index=True) if curve_tables else pd.DataFrame()
    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    return curves, summary_df.sort_values("search_name").reset_index(drop=True)


class FullScrapeDataCapTests(unittest.TestCase):
    def test_recommendation_logic_marks_clear_plateau(self):
        table = pd.DataFrame(
            [
                {"search_name": "gucci", "approach": "numeric_tree_v1", "fraction": 0.2, "test_pr_auc_mean": 0.70, "test_precision_mean": 0.78, "test_precision_at_10_mean": 0.82},
                {"search_name": "gucci", "approach": "numeric_tree_v1", "fraction": 0.8, "test_pr_auc_mean": 0.81, "test_precision_mean": 0.90, "test_precision_at_10_mean": 0.92},
                {"search_name": "gucci", "approach": "numeric_tree_v1", "fraction": 1.0, "test_pr_auc_mean": 0.812, "test_precision_mean": 0.905, "test_precision_at_10_mean": 0.92},
            ]
        )
        summary = recommendation_from_curve(table)
        self.assertEqual(summary.recommendation, "near_cap")


def main() -> None:
    curves, summary = run_analysis()
    payload = {
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "curves": curves.replace({np.nan: None}).to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-analysis":
        sys.argv = [sys.argv[0]]
        main()
    else:
        unittest.main()