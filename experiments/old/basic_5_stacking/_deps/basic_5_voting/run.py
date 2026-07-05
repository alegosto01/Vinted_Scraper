#!/usr/bin/env python3
from __future__ import annotations

import argparse
import __main__
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.basic_5_stacking._deps.basic_5_voting.paths import (
    EXPERIMENT_ROOT,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)
from experiments.old.basic_5_stacking._deps.deal_finder import model_sweep as base_sweep
from experiments.old.basic_5_stacking._deps.deal_finder.model_sweep import RulePriceScorer
from experiments.old.basic_5_stacking._deps.deal_finder.modeling import TARGET_COL, load_pickle, score_with_model
from experiments.old.basic_5_stacking._deps.full_scrape_model.compare_feature_modalities import add_full_engineered_features
from experiments.old.basic_5_stacking._deps.full_scrape_model.dataset import TASK_SOLD_STATUS, build_search_dataset
from experiments.old.basic_5_stacking._deps.full_scrape_model.paths import MODELS_DIR


setattr(__main__, "RulePriceScorer", RulePriceScorer)


DEFAULT_ORIGINAL_RUN = "sold_status_feature_modalities_20260515_full_visual"
DEFAULT_NEW_RUN = "sold_status_basic_extra_models_20260521"
DEFAULT_SEARCHES = (
    "nike",
    "ps4",
    "gucci",
    "prada",
    "griffati_uomo_all",
    "griffati_donna_all",
)
ORIGINAL_APPROACHES = (
    "logistic_v1_baseline",
    "logistic_snapshot_v2",
    "sgd_text_numeric_v1",
    "linear_svm_calibrated_v1",
    "numeric_tree_v1",
    "rules_price_v1",
)
NEW_APPROACHES = (
    "random_forest_basic_v1",
    "hist_gradient_basic_numeric_v1",
    "xgboost_basic_v1",
)
ID_COLUMNS = ["item_id", "Dataid", "Title", "Brand", "Size", "Price", "Likes", "Link"]


def numeric(value: object) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else float("nan")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y_true)) < 2:
            return float("nan"), float("nan")
        return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan"), float("nan")


def safe_kappa(left: np.ndarray, right: np.ndarray) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(left, right))
    except Exception:
        return float("nan")


def threshold_accuracy(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    keep = np.isfinite(values) & np.isfinite(labels)
    if not keep.any() or not np.isfinite(threshold):
        return float("nan")
    return float(((values[keep] >= threshold).astype(int) == labels[keep]).mean())


def tune_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    keep = np.isfinite(values) & np.isfinite(labels)
    if not keep.any():
        return float("nan"), float("nan")

    labels = labels[keep]
    values = values[keep]
    order = np.argsort(-values, kind="mergesort")
    labels = labels[order]
    values = values[order]

    correct = int((labels == 0).sum())
    best_correct = correct
    best_threshold = float(np.nextafter(values[0], np.inf))
    index = 0
    while index < len(values):
        next_index = index + 1
        while next_index < len(values) and values[next_index] == values[index]:
            next_index += 1
        group = labels[index:next_index]
        correct += int((group == 1).sum()) - int((group == 0).sum())
        # Keep the stricter threshold on an accuracy tie.
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(values[index])
        index = next_index
    return best_threshold, safe_divide(best_correct, len(values))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_path(run_name: str, search: str, approach: str, seed: int) -> Path:
    return MODELS_DIR / f"{run_name}_{search}_basic_5_{approach}_seed{seed}_metadata.json"


def load_metadata(run_name: str, search: str, approaches: tuple[str, ...], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for approach in approaches:
        path = metadata_path(run_name, search, approach, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata for {search}/{approach}: {path}")
        metadata = read_json(path)
        artifact = Path(str(metadata.get("artifact_path", "")))
        if not artifact.exists():
            raise FileNotFoundError(f"Missing model artifact for {search}/{approach}: {artifact}")
        metadata["metadata_path"] = str(path)
        metadata["artifact_path"] = str(artifact)
        metadata["approach"] = str(metadata.get("approach", approach))
        rows.append(metadata)
    return rows


def common_split_frames(search: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = build_search_dataset(TASK_SOLD_STATUS, search)
    frame = base_sweep.prepare_sweep_frame(frame)
    frame = add_full_engineered_features(frame)
    splits = base_sweep.stratified_random_split(frame, seed=seed)
    return splits.validation.reset_index(drop=True), splits.test.reset_index(drop=True)


def score_search(
    search: str,
    *,
    seed: int,
    original_run: str,
    new_run: str,
    threshold_objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation, frame = common_split_frames(search, seed)
    if validation.empty:
        raise ValueError(f"Common validation frame is empty for {search}")
    if frame.empty:
        raise ValueError(f"Common test frame is empty for {search}")
    metadata = [
        *load_metadata(original_run, search, ORIGINAL_APPROACHES, seed),
        *load_metadata(new_run, search, NEW_APPROACHES, seed),
    ]
    validation_sold = pd.to_numeric(validation[TARGET_COL], errors="coerce").fillna(0).astype(int)
    sold = pd.to_numeric(frame[TARGET_COL], errors="coerce").fillna(0).astype(int)
    keep = [col for col in ID_COLUMNS if col in frame.columns]
    item_votes = frame[keep].copy()
    item_votes["search_name"] = search
    item_votes["offline_sold_label"] = sold.to_numpy()
    model_rows: list[dict[str, Any]] = []
    vote_cols: list[str] = []
    score_cols: list[str] = []
    for model_metadata in metadata:
        approach = str(model_metadata["approach"])
        model = load_pickle(Path(model_metadata["artifact_path"]))
        validation_scores = np.asarray(score_with_model(model, validation), dtype=float)
        validation_scores = np.clip(validation_scores, 0.0, 1.0)
        scores = np.asarray(score_with_model(model, frame), dtype=float)
        scores = np.clip(scores, 0.0, 1.0)
        saved_threshold = numeric(model_metadata.get("threshold"))
        if threshold_objective == "accuracy":
            threshold, validation_accuracy = tune_accuracy_threshold(
                validation_sold.to_numpy(),
                validation_scores,
            )
        elif threshold_objective == "saved_precision":
            threshold = saved_threshold
            validation_accuracy = threshold_accuracy(validation_sold.to_numpy(), validation_scores, threshold)
        else:
            raise ValueError(f"Unsupported threshold objective: {threshold_objective}")
        votes = scores >= threshold
        score_col = f"score__{approach}"
        vote_col = f"vote__{approach}"
        item_votes[score_col] = scores
        item_votes[vote_col] = votes.astype(int)
        vote_cols.append(vote_col)
        score_cols.append(score_col)
        roc_auc, pr_auc = safe_auc(sold.to_numpy(), scores)
        model_rows.append(
            {
                "search_name": search,
                "approach": approach,
                "threshold_objective": threshold_objective,
                "saved_precision_threshold": saved_threshold,
                "threshold": threshold,
                "validation_items": int(len(validation)),
                "validation_accuracy": validation_accuracy,
                "test_items": int(len(frame)),
                "test_accuracy": threshold_accuracy(sold.to_numpy(), scores, threshold),
                "sold_base_rate": float(sold.mean()),
                "sold_vote_count": int(votes.sum()),
                "sold_vote_rate": safe_divide(int(votes.sum()), len(frame)),
                "sold_vote_precision": safe_divide(int(sold[votes].sum()), int(votes.sum())),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "metadata_path": str(model_metadata.get("metadata_path", "")),
            }
        )
    vote_matrix = item_votes[vote_cols].to_numpy(dtype=int)
    score_matrix = item_votes[score_cols].to_numpy(dtype=float)
    item_votes["sold_vote_count"] = vote_matrix.sum(axis=1)
    item_votes["sold_vote_fraction"] = item_votes["sold_vote_count"] / len(vote_cols)
    item_votes["mean_model_score"] = np.nanmean(score_matrix, axis=1)
    item_votes["majority_sold_vote"] = item_votes["sold_vote_count"] >= (len(vote_cols) // 2 + 1)
    item_votes["any_sold_vote"] = item_votes["sold_vote_count"] > 0
    item_votes["all_models_agree"] = item_votes["sold_vote_count"].isin((0, len(vote_cols)))
    item_votes["all_models_vote_sold"] = item_votes["sold_vote_count"].eq(len(vote_cols))
    return item_votes, pd.DataFrame(model_rows)


def pairwise_agreement(item_votes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    vote_cols = [col for col in item_votes.columns if col.startswith("vote__")]
    for search, group in item_votes.groupby("search_name", sort=True):
        for left_col, right_col in combinations(vote_cols, 2):
            left = group[left_col].astype(int).to_numpy()
            right = group[right_col].astype(int).to_numpy()
            both_positive = int(np.logical_and(left == 1, right == 1).sum())
            either_positive = int(np.logical_or(left == 1, right == 1).sum())
            rows.append(
                {
                    "search_name": search,
                    "model_a": left_col.removeprefix("vote__"),
                    "model_b": right_col.removeprefix("vote__"),
                    "agreement_rate": float((left == right).mean()),
                    "cohen_kappa": safe_kappa(left, right),
                    "positive_jaccard": safe_divide(both_positive, either_positive),
                    "both_vote_sold_count": both_positive,
                    "either_votes_sold_count": either_positive,
                }
            )
    return pd.DataFrame(rows)


def vote_count_distribution(item_votes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (search, vote_count), group in item_votes.groupby(["search_name", "sold_vote_count"], sort=True):
        rows.append(
            {
                "search_name": search,
                "sold_vote_count": int(vote_count),
                "items": int(len(group)),
                "item_fraction": safe_divide(len(group), len(item_votes[item_votes["search_name"] == search])),
                "sold_items": int(group["offline_sold_label"].eq(1).sum()),
                "sold_rate": float(group["offline_sold_label"].mean()),
            }
        )
    return pd.DataFrame(rows)


def ensemble_summary(
    item_votes: pd.DataFrame,
    model_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for search, group in item_votes.groupby("search_name", sort=True):
        sold = group["offline_sold_label"].astype(int).to_numpy()
        vote_auc, vote_pr_auc = safe_auc(sold, group["sold_vote_fraction"].to_numpy(dtype=float))
        mean_auc, mean_pr_auc = safe_auc(sold, group["mean_model_score"].to_numpy(dtype=float))
        majority_auc, majority_pr_auc = safe_auc(sold, group["majority_sold_vote"].astype(int).to_numpy())
        pair_group = pairwise[pairwise["search_name"] == search]
        model_group = model_summary[model_summary["search_name"] == search]
        rows.append(
            {
                "search_name": search,
                "test_items": int(len(group)),
                "sold_base_rate": float(group["offline_sold_label"].mean()),
                "models": int(len([col for col in group.columns if col.startswith("vote__")])),
                "vote_fraction_roc_auc": vote_auc,
                "vote_fraction_pr_auc": vote_pr_auc,
                "majority_vote_roc_auc": majority_auc,
                "majority_vote_pr_auc": majority_pr_auc,
                "mean_score_roc_auc": mean_auc,
                "mean_score_pr_auc": mean_pr_auc,
                "best_individual_roc_auc": float(model_group["roc_auc"].max()),
                "best_individual_pr_auc": float(model_group["pr_auc"].max()),
                "majority_sold_count": int(group["majority_sold_vote"].sum()),
                "any_sold_vote_count": int(group["any_sold_vote"].sum()),
                "unanimous_sold_count": int(group["all_models_vote_sold"].sum()),
                "all_models_agree_rate": float(group["all_models_agree"].mean()),
                "mean_pairwise_agreement": float(pair_group["agreement_rate"].mean()),
                "mean_pairwise_kappa": float(pair_group["cohen_kappa"].mean()),
                "mean_positive_jaccard": float(pair_group["positive_jaccard"].mean()),
            }
        )
    return pd.DataFrame(rows)


def fmt_number(value: object, digits: int = 3) -> str:
    number = numeric(value)
    return "" if pd.isna(number) else f"{number:.{digits}f}"


def fmt_pct(value: object) -> str:
    number = numeric(value)
    return "" if pd.isna(number) else f"{100.0 * number:.1f}%"


def fmt_int(value: object) -> str:
    number = numeric(value)
    return "" if pd.isna(number) else str(int(number))


def write_table(frame: pd.DataFrame, cols: list[tuple[str, str]], kinds: dict[str, str]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _col, label in cols) + " |",
        "| " + " | ".join("---" if col == "search_name" else "---:" for col, _label in cols) + " |",
    ]
    for _idx, row in frame.iterrows():
        values = []
        for col, _label in cols:
            value = row.get(col)
            kind = kinds.get(col, "number")
            if col == "search_name":
                values.append(f"`{value}`")
            elif kind == "pct":
                values.append(fmt_pct(value))
            elif kind == "int":
                values.append(fmt_int(value))
            elif kind == "text":
                values.append(f"`{value}`")
            else:
                values.append(fmt_number(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    threshold_objective: str,
) -> Path:
    path = assert_experiment_path(out_dir / "basic_5_voting_report.md")
    if threshold_objective == "accuracy":
        threshold_line = (
            "A model votes `sold` when its score is at or above the threshold re-selected on the "
            "validation split to maximize sold/not-sold accuracy."
        )
        agreement_note = (
            "`Pairwise agree` is agreement on hard sold/not-sold votes after validation-accuracy "
            "threshold tuning. Positive-selection Jaccard shows whether models select the same sold items."
        )
    else:
        threshold_line = (
            "A model votes `sold` when its score is at or above the saved validation-selected "
            "precision threshold from its training sweep."
        )
        agreement_note = (
            "`Pairwise agree` is agreement on hard sold/not-sold votes. Saved precision-first "
            "thresholds can make sold votes sparse, so positive-selection Jaccard is the sharper "
            "selection-overlap check."
        )
    auc_cols = [
        ("search_name", "Search"),
        ("vote_fraction_roc_auc", "Vote AUC"),
        ("vote_fraction_pr_auc", "Vote PR AUC"),
        ("majority_vote_roc_auc", "Majority AUC"),
        ("mean_score_roc_auc", "Mean-score AUC"),
        ("best_individual_roc_auc", "Best individual AUC"),
    ]
    agreement_cols = [
        ("search_name", "Search"),
        ("test_items", "Test items"),
        ("majority_sold_count", "Majority sold"),
        ("any_sold_vote_count", "Any sold"),
        ("unanimous_sold_count", "All sold"),
        ("all_models_agree_rate", "All agree"),
        ("mean_pairwise_agreement", "Pairwise agree"),
        ("mean_pairwise_kappa", "Mean kappa"),
        ("mean_positive_jaccard", "Positive Jaccard"),
    ]
    best_models = (
        model_summary.sort_values(["search_name", "roc_auc"], ascending=[True, False])
        .drop_duplicates("search_name", keep="first")
        .reset_index(drop=True)
    )
    best_cols = [
        ("search_name", "Search"),
        ("approach", "Best individual"),
        ("roc_auc", "ROC AUC"),
        ("pr_auc", "PR AUC"),
        ("threshold", "Vote threshold"),
        ("validation_accuracy", "Val accuracy"),
        ("test_accuracy", "Test accuracy"),
        ("sold_vote_count", "Sold votes"),
        ("sold_vote_precision", "Vote precision"),
    ]
    kinds = {
        "test_items": "int",
        "majority_sold_count": "int",
        "any_sold_vote_count": "int",
        "unanimous_sold_count": "int",
        "sold_vote_count": "int",
        "approach": "text",
        "all_models_agree_rate": "pct",
        "mean_pairwise_agreement": "pct",
        "mean_positive_jaccard": "pct",
        "validation_accuracy": "pct",
        "test_accuracy": "pct",
        "sold_vote_precision": "pct",
    }
    lines = [
        "# Basic 5 Voting Ensemble",
        "",
        "This experiment scores the same held-out offline sold-status test rows with nine `basic_5` models.",
        "The hard-vote models see only `Title`, `Brand`, `Size`, `Price`, and `Likes`.",
        "",
        "- Six original non-visual families: logistic baseline, logistic snapshot, SGD, calibrated linear SVM, numeric tree, and price rules.",
        "- Three added families: random forest, histogram gradient numeric booster, and XGBoost.",
        f"- {threshold_line}",
        "- `Vote AUC` uses the fraction of sold votes from 0/9 through 9/9 as the ensemble ranking score.",
        "- `Majority AUC` uses the binary majority vote only; it throws away vote-count ordering.",
        "- `Mean-score AUC` averages model scores as a soft-vote reference.",
        "- Individual model ROC AUC is score-ranking based and does not change when a hard-vote threshold changes.",
        "",
        "## Ensemble AUC",
        "",
        *write_table(summary, auc_cols, kinds),
        "",
        "## Agreement",
        "",
        agreement_note,
        "",
        *write_table(summary, agreement_cols, kinds),
        "",
        "## Best Individual ROC AUC",
        "",
        *write_table(best_models, best_cols, kinds),
        "",
        "## Files",
        "",
        "- `item_votes.csv`: common held-out rows, every model score and vote, ensemble vote count, and labels.",
        "- `ensemble_summary.csv`: ensemble AUC and agreement summaries by search.",
        "- `model_summary.csv`: individual model AUC and sold-vote rates.",
        "- `pairwise_agreement.csv`: pairwise hard-vote agreement, Cohen kappa, and positive-selection Jaccard.",
        "- `vote_count_distribution.csv`: item counts and sold rates for each 0-to-9 sold-vote bucket.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hard-voting ensembles across nine basic_5 sold-status models.")
    parser.add_argument("--original-run", default=DEFAULT_ORIGINAL_RUN)
    parser.add_argument("--new-run", default=DEFAULT_NEW_RUN)
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    parser.add_argument(
        "--threshold-objective",
        choices=("accuracy", "saved_precision"),
        default="accuracy",
        help="Retune hard-vote thresholds for validation accuracy or reuse saved precision-first sweep thresholds.",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    searches = list(dict.fromkeys(args.search or DEFAULT_SEARCHES))
    out_dir = assert_experiment_path(
        Path(args.out_dir) if args.out_dir else EXPERIMENT_ROOT / run_id("basic_5_voting")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    item_parts: list[pd.DataFrame] = []
    model_parts: list[pd.DataFrame] = []
    for search in searches:
        item_votes, model_summary = score_search(
            search,
            seed=args.seed,
            original_run=args.original_run,
            new_run=args.new_run,
            threshold_objective=args.threshold_objective,
        )
        item_parts.append(item_votes)
        model_parts.append(model_summary)
        print(f"[basic_5_voting] search={search} test_rows={len(item_votes)} models={len(model_summary)}", flush=True)
    items = pd.concat(item_parts, ignore_index=True)
    model_summary = pd.concat(model_parts, ignore_index=True)
    pairwise = pairwise_agreement(items)
    distribution = vote_count_distribution(items)
    summary = ensemble_summary(items, model_summary, pairwise)
    items.to_csv(assert_experiment_path(out_dir / "item_votes.csv"), index=False)
    model_summary.to_csv(assert_experiment_path(out_dir / "model_summary.csv"), index=False)
    pairwise.to_csv(assert_experiment_path(out_dir / "pairwise_agreement.csv"), index=False)
    distribution.to_csv(assert_experiment_path(out_dir / "vote_count_distribution.csv"), index=False)
    summary.to_csv(assert_experiment_path(out_dir / "ensemble_summary.csv"), index=False)
    report_path = write_report(out_dir, summary, model_summary, args.threshold_objective)
    write_manifest(
        out_dir / "manifest.json",
        command="basic_5_voting.run",
        extra={
            "original_run": args.original_run,
            "new_run": args.new_run,
            "searches": searches,
            "seed": args.seed,
            "threshold_objective": args.threshold_objective,
            "approaches": [*ORIGINAL_APPROACHES, *NEW_APPROACHES],
            "outputs": {
                "report": str(report_path),
                "ensemble_summary": str(out_dir / "ensemble_summary.csv"),
                "model_summary": str(out_dir / "model_summary.csv"),
                "item_votes": str(out_dir / "item_votes.csv"),
                "pairwise_agreement": str(out_dir / "pairwise_agreement.csv"),
                "vote_count_distribution": str(out_dir / "vote_count_distribution.csv"),
            },
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
