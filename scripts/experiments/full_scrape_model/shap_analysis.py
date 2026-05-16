#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep
from experiments.deal_finder.modeling import TARGET_COL, load_pickle, score_with_model
from experiments.full_scrape_model.compare_feature_modalities import (
    DEFAULT_EXCLUDED_SEARCHES,
    add_full_engineered_features,
)
from experiments.full_scrape_model.paths import (
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)


DEFAULT_MODALITY_RUN = "sold_status_feature_modalities_20260515_full_visual"
DEFAULT_MODES = ("basic_5", "full_scrape_plus_visual")
IDENTITY_COLUMNS = ["SearchName", "item_id", "Dataid", "Link", "Title", "Brand", "Size", "Price", "Likes"]


@dataclass(frozen=True)
class ShapBundle:
    values: np.ndarray
    expected_value: float | None
    method: str


def run_dir_for_name(run_name: str) -> Path:
    path = OFFLINE_RUNS_DIR / run_name
    if not path.exists():
        raise FileNotFoundError(f"Run folder not found: {path}")
    return path


def metadata_for_row(row: pd.Series, *, modality_run: str) -> dict[str, Any]:
    search = str(row["search_name"])
    mode = str(row["feature_mode"])
    approach = str(row["approach"])
    seed = int(row.get("seed", 42))
    pattern = f"{modality_run}_{search}_{mode}_{approach}_seed{seed}_metadata.json"
    matches = sorted(MODELS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No model metadata matched {pattern}")
    metadata = json.loads(matches[-1].read_text(encoding="utf-8"))
    artifact = Path(str(metadata.get("artifact_path", "")))
    if not artifact.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact}")
    metadata["metadata_path"] = str(matches[-1])
    metadata["artifact_path"] = str(artifact)
    return metadata


def clean_transformed_feature_name(name: object) -> tuple[str, str, str]:
    text = str(name)
    if text.startswith("numeric__"):
        original = text.replace("numeric__", "", 1)
        return original, original, feature_group(original)
    if text.startswith("text_") and "__" in text:
        prefix, token = text.split("__", 1)
        original = prefix.replace("text_", "", 1)
        return original, f"{original}: {token}", "text"
    return text, text, feature_group(text)


def feature_group(original_feature: str) -> str:
    text = str(original_feature)
    if text.startswith("DinoEmbedding_"):
        return "dino_embedding"
    if text in {"Title", "Brand", "Size", "Description", "Condition", "Upload_date", "SellerName", "Location"}:
        return "text"
    if text.startswith("Photo") or text in {
        "SimpleBadPhotoScore",
        "PyiqaQualityScore",
        "PyiqaBadPhotoScore",
        "AestheticGoodScore",
        "AestheticBadPhotoScore",
        "DinoEmbeddingDim",
        "DinoEmbeddingNorm",
        "DinoOutlierScore",
        "CombinedBadPhotoScore",
        "PictureCountNum",
    }:
        return "visual_readable"
    if text in {
        "Interested_count",
        "View_count",
        "ReviewsCount",
        "Stars",
        "PictureCount",
        "VisiblePictureCount",
        "HiddenPictureCount",
        "Upload_date_days",
        "description_char_len",
        "description_token_count",
        "condition_present",
        "seller_present",
        "location_present",
        "has_reviews",
        "has_stars",
    }:
        return "full_scrape"
    return "basic_numeric"


def is_dino_embedding_feature(feature_name: object) -> bool:
    text = str(feature_name)
    return "DinoEmbedding_" in text


def transformed_matrix(model: Any, frame: pd.DataFrame) -> tuple[Any, list[str]]:
    if not hasattr(model, "named_steps") or "features" not in model.named_steps:
        raise TypeError("SHAP analysis expects a sklearn Pipeline with a 'features' step.")
    preprocessor = model.named_steps["features"]
    matrix = preprocessor.transform(frame)
    try:
        names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names = fallback_feature_names(preprocessor)
    return matrix, names


def fallback_feature_names(preprocessor: Any) -> list[str]:
    names: list[str] = []
    for name, transformer, cols in getattr(preprocessor, "transformers_", []):
        if name == "remainder" or transformer == "drop":
            continue
        if name == "numeric":
            numeric_cols = list(cols)
            imputer = getattr(transformer, "named_steps", {}).get("imputer") if hasattr(transformer, "named_steps") else None
            statistics = getattr(imputer, "statistics_", None)
            if statistics is not None and len(statistics) == len(numeric_cols):
                numeric_cols = [col for col, stat in zip(numeric_cols, statistics) if not pd.isna(stat)]
            names.extend([f"numeric__{col}" for col in numeric_cols])
            continue
        if str(name).startswith("text_"):
            source_col = str(name).replace("text_", "", 1)
            try:
                tfidf = transformer.named_steps["tfidf"]
                names.extend([f"{name}__{term}" for term in tfidf.get_feature_names_out()])
            except Exception:
                names.append(f"text_{source_col}__unknown")
            continue
    return names


def to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def final_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def calibrated_linear_coef(estimator: Any) -> np.ndarray | None:
    classifiers = getattr(estimator, "calibrated_classifiers_", None)
    if not classifiers:
        return None
    coefs = []
    for classifier in classifiers:
        base = getattr(classifier, "estimator", None) or getattr(classifier, "base_estimator", None)
        coef = getattr(base, "coef_", None)
        if coef is not None:
            coefs.append(np.asarray(coef).reshape(-1))
    if not coefs:
        return None
    return np.mean(np.vstack(coefs), axis=0)


def linear_coef(estimator: Any) -> np.ndarray | None:
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        return np.asarray(coef).reshape(-1)
    return calibrated_linear_coef(estimator)


def explain_transformed(model: Any, background: np.ndarray, sample: np.ndarray) -> ShapBundle:
    estimator = final_estimator(model)
    coef = linear_coef(estimator)
    if coef is not None:
        center = np.nanmean(background, axis=0)
        values = (sample - center) * coef
        expected = float(np.nanmean(background @ coef)) if background.size else None
        return ShapBundle(values=np.asarray(values, dtype=float), expected_value=expected, method="linear_exact_margin")

    if estimator.__class__.__name__.lower().startswith("extratrees"):
        try:
            import shap
        except Exception as exc:
            raise RuntimeError("Tree SHAP requires the optional 'shap' package. Install shap to explain tree models.") from exc
        explainer = shap.TreeExplainer(estimator)
        raw_values = explainer.shap_values(sample)
        if isinstance(raw_values, list):
            values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
        elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
            values = raw_values[:, :, 1] if raw_values.shape[-1] > 1 else raw_values[:, :, 0]
        else:
            values = raw_values
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected_value = float(np.asarray(expected).reshape(-1)[-1])
        else:
            expected_value = float(expected)
        return ShapBundle(values=np.asarray(values, dtype=float), expected_value=expected_value, method="tree_shap")

    raise RuntimeError(f"No SHAP implementation is available for estimator {estimator.__class__.__name__}")


def prepare_frame(dataset_path: Path, metadata: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(dataset_path, low_memory=False)
    work = add_full_engineered_features(frame)
    for col in metadata.get("numeric_features", []) or []:
        if col not in work.columns:
            work[col] = np.nan
    for col in metadata.get("text_features", []) or []:
        if col not in work.columns:
            work[col] = ""
    splits = base_sweep.stratified_random_split(work, seed=seed)
    return splits.train, splits.validation, splits.test


def choose_rows(frame: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def feature_importance_frame(
    *,
    values: np.ndarray,
    feature_names: list[str],
    search_name: str,
    feature_mode: str,
    approach: str,
    explainer_method: str,
    exclude_dino_embeddings: bool,
) -> pd.DataFrame:
    rows = []
    for idx, transformed in enumerate(feature_names):
        if exclude_dino_embeddings and is_dino_embedding_feature(transformed):
            continue
        original, display, group = clean_transformed_feature_name(transformed)
        col_values = values[:, idx]
        rows.append(
            {
                "search_name": search_name,
                "feature_mode": feature_mode,
                "approach": approach,
                "explainer_method": explainer_method,
                "transformed_feature": transformed,
                "original_feature": original,
                "display_feature": display,
                "feature_group": group,
                "mean_abs_shap": float(np.nanmean(np.abs(col_values))),
                "mean_shap": float(np.nanmean(col_values)),
                "positive_rate": float(np.nanmean(col_values > 0)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def item_explanations_frame(
    *,
    sample: pd.DataFrame,
    values: np.ndarray,
    feature_names: list[str],
    scores: np.ndarray,
    search_name: str,
    feature_mode: str,
    approach: str,
    top_n: int,
    exclude_dino_embeddings: bool,
) -> pd.DataFrame:
    rows = []
    usable_indices = [idx for idx, name in enumerate(feature_names) if not (exclude_dino_embeddings and is_dino_embedding_feature(name))]
    for row_pos, (_, item) in enumerate(sample.reset_index(drop=True).iterrows()):
        item_values = values[row_pos, usable_indices]
        ordered = np.argsort(-np.abs(item_values))[:top_n]
        positive = []
        negative = []
        for rel_idx in ordered:
            feature_idx = usable_indices[int(rel_idx)]
            original, display, group = clean_transformed_feature_name(feature_names[feature_idx])
            payload = {
                "feature": display,
                "original_feature": original,
                "group": group,
                "shap_value": float(values[row_pos, feature_idx]),
            }
            if payload["shap_value"] >= 0:
                positive.append(payload)
            else:
                negative.append(payload)
        identity = {col: item.get(col) for col in IDENTITY_COLUMNS if col in item.index}
        rows.append(
            {
                "search_name": search_name,
                "feature_mode": feature_mode,
                "approach": approach,
                **identity,
                "score": float(scores[row_pos]) if row_pos < len(scores) else np.nan,
                "label": int(item.get(TARGET_COL, 0)) if pd.notna(item.get(TARGET_COL, np.nan)) else np.nan,
                "top_positive_drivers": json.dumps(positive[:top_n], ensure_ascii=True),
                "top_negative_drivers": json.dumps(negative[:top_n], ensure_ascii=True),
            }
        )
    return pd.DataFrame(rows)


def analyze_one(
    *,
    run_dir: Path,
    row: pd.Series,
    out_dir: Path,
    max_background_rows: int,
    max_explain_rows: int,
    top_item_features: int,
    exclude_dino_embeddings: bool,
) -> dict[str, Any]:
    metadata = metadata_for_row(row, modality_run=run_dir.name)
    search_name = str(metadata["search_name"])
    feature_mode = str(metadata["feature_mode"])
    approach = str(metadata["approach"])
    seed = int(metadata.get("seed", 42))
    dataset_path = run_dir / "datasets" / f"{search_name}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found for {search_name}: {dataset_path}")

    model = load_pickle(Path(metadata["artifact_path"]))
    train, _validation, test = prepare_frame(dataset_path, metadata, seed)
    background = choose_rows(train, max_rows=max_background_rows, seed=seed)
    sample = choose_rows(test, max_rows=max_explain_rows, seed=seed + 7)
    if background.empty or sample.empty:
        raise RuntimeError(f"Cannot explain {search_name}/{feature_mode}: empty background or sample")

    background_matrix, feature_names = transformed_matrix(model, background)
    sample_matrix, sample_feature_names = transformed_matrix(model, sample)
    if sample_feature_names != feature_names:
        raise RuntimeError(f"Feature names changed between background and sample for {search_name}/{feature_mode}")

    background_dense = to_dense(background_matrix)
    sample_dense = to_dense(sample_matrix)
    bundle = explain_transformed(model, background_dense, sample_dense)
    scores = score_with_model(model, sample)

    importance = feature_importance_frame(
        values=bundle.values,
        feature_names=feature_names,
        search_name=search_name,
        feature_mode=feature_mode,
        approach=approach,
        explainer_method=bundle.method,
        exclude_dino_embeddings=exclude_dino_embeddings,
    )
    item_explanations = item_explanations_frame(
        sample=sample,
        values=bundle.values,
        feature_names=feature_names,
        scores=scores,
        search_name=search_name,
        feature_mode=feature_mode,
        approach=approach,
        top_n=top_item_features,
        exclude_dino_embeddings=exclude_dino_embeddings,
    )

    prefix = f"{search_name}_{feature_mode}_{approach}"
    importance_path = out_dir / f"{prefix}_feature_importance.csv"
    item_path = out_dir / f"{prefix}_item_explanations.csv"
    importance.to_csv(importance_path, index=False)
    item_explanations.to_csv(item_path, index=False)
    return {
        "search_name": search_name,
        "feature_mode": feature_mode,
        "approach": approach,
        "explainer_method": bundle.method,
        "artifact_path": metadata["artifact_path"],
        "metadata_path": metadata["metadata_path"],
        "background_rows": int(len(background)),
        "explained_rows": int(len(sample)),
        "transformed_feature_count": int(len(feature_names)),
        "reported_feature_count": int(len(importance)),
        "excluded_dino_embedding_features": int(sum(is_dino_embedding_feature(name) for name in feature_names)) if exclude_dino_embeddings else 0,
        "importance_path": str(importance_path),
        "item_explanations_path": str(item_path),
    }


def write_report(out_dir: Path, summary: pd.DataFrame, combined_importance: pd.DataFrame, group_importance: pd.DataFrame) -> Path:
    report_path = out_dir / "shap_analysis_report.md"
    lines = [
        "# SHAP Analysis Without DINO Embedding Dimensions",
        "",
        f"Run folder: `{out_dir}`",
        "",
        "This report explains the already-trained `basic_5` and `full_scrape_plus_visual` models.",
        "DINO embedding dimensions such as `DinoEmbedding_0000` are excluded from the reported SHAP tables.",
        "Readable DINO summary features such as `DinoEmbeddingNorm` and `DinoOutlierScore` are kept when present.",
        "",
        "## Models Explained",
        "",
        "| search | mode | approach | method | explained rows | reported features | excluded DINO dims |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["search_name", "feature_mode"]).iterrows():
        lines.append(
            f"| {row.get('search_name')} | {row.get('feature_mode')} | {row.get('approach')} | "
            f"{row.get('explainer_method')} | {int(row.get('explained_rows', 0))} | "
            f"{int(row.get('reported_feature_count', 0))} | {int(row.get('excluded_dino_embedding_features', 0))} |"
        )
    if not group_importance.empty:
        lines.extend(["", "## Top Feature Groups", ""])
        top_groups = group_importance.sort_values("mean_abs_shap", ascending=False).head(25)
        lines.extend(["| search | mode | group | mean abs SHAP |", "| --- | --- | --- | ---: |"])
        for _, row in top_groups.iterrows():
            lines.append(
                f"| {row.get('search_name')} | {row.get('feature_mode')} | {row.get('feature_group')} | "
                f"{float(row.get('mean_abs_shap', 0.0)):.6f} |"
            )
    if not combined_importance.empty:
        lines.extend(["", "## Top Readable Features", ""])
        top = combined_importance.sort_values("mean_abs_shap", ascending=False).head(40)
        lines.extend(["| search | mode | feature | group | mean abs SHAP | mean SHAP |", "| --- | --- | --- | --- | ---: | ---: |"])
        for _, row in top.iterrows():
            lines.append(
                f"| {row.get('search_name')} | {row.get('feature_mode')} | {row.get('display_feature')} | "
                f"{row.get('feature_group')} | {float(row.get('mean_abs_shap', 0.0)):.6f} | "
                f"{float(row.get('mean_shap', 0.0)):.6f} |"
            )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain full-scrape feature-modality models with SHAP-style values.")
    parser.add_argument("--modality-run", default=DEFAULT_MODALITY_RUN)
    parser.add_argument("--mode", action="append", default=[])
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--include-excluded-searches", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-background-rows", type=int, default=120)
    parser.add_argument("--max-explain-rows", type=int, default=180)
    parser.add_argument("--top-item-features", type=int, default=8)
    parser.add_argument("--include-dino-embedding-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    run_dir = run_dir_for_name(args.modality_run)
    modes = tuple(args.mode or DEFAULT_MODES)
    best_path = run_dir / "best_by_search_mode.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"best_by_search_mode.csv not found: {best_path}")
    best = pd.read_csv(best_path)
    best = best[best["feature_mode"].astype(str).isin(modes)].copy()
    if not args.include_excluded_searches:
        best = best[~best["search_name"].astype(str).isin(DEFAULT_EXCLUDED_SEARCHES)].copy()
    if args.search:
        wanted = {value.lower() for value in args.search}
        best = best[best["search_name"].astype(str).str.lower().isin(wanted)].copy()
    if best.empty:
        raise SystemExit("No matching model rows found to explain.")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "shap_analysis" / run_id("no_dino")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary_rows = []
    importance_frames = []
    item_frames = []
    for _, row in best.sort_values(["search_name", "feature_mode"]).iterrows():
        result = analyze_one(
            run_dir=run_dir,
            row=row,
            out_dir=out_dir,
            max_background_rows=max(1, int(args.max_background_rows)),
            max_explain_rows=max(1, int(args.max_explain_rows)),
            top_item_features=max(1, int(args.top_item_features)),
            exclude_dino_embeddings=not bool(args.include_dino_embedding_features),
        )
        summary_rows.append(result)
        importance_frames.append(pd.read_csv(result["importance_path"]))
        item_frames.append(pd.read_csv(result["item_explanations_path"]))

    summary = pd.DataFrame(summary_rows)
    combined_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    combined_items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
    group_importance = (
        combined_importance.groupby(["search_name", "feature_mode", "approach", "feature_group"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), feature_count=("transformed_feature", "count"))
        if not combined_importance.empty
        else pd.DataFrame()
    )

    summary_path = out_dir / "shap_model_summary.csv"
    importance_path = out_dir / "shap_feature_importance_long.csv"
    group_path = out_dir / "shap_group_importance.csv"
    item_path = out_dir / "shap_item_explanations.csv"
    summary.to_csv(summary_path, index=False)
    combined_importance.to_csv(importance_path, index=False)
    group_importance.to_csv(group_path, index=False)
    combined_items.to_csv(item_path, index=False)
    report_path = write_report(out_dir, summary, combined_importance, group_importance)

    manifest = {
        "created_at": utc_now_iso(),
        "modality_run": args.modality_run,
        "modes": list(modes),
        "searches": args.search or "all",
        "include_excluded_searches": bool(args.include_excluded_searches),
        "max_background_rows": int(args.max_background_rows),
        "max_explain_rows": int(args.max_explain_rows),
        "top_item_features": int(args.top_item_features),
        "include_dino_embedding_features": bool(args.include_dino_embedding_features),
        "elapsed_seconds": float(time.perf_counter() - started),
        "summary_path": str(summary_path),
        "importance_path": str(importance_path),
        "group_importance_path": str(group_path),
        "item_explanations_path": str(item_path),
        "report_path": str(report_path),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_manifest(out_dir / "run_manifest.json", command="full_scrape_model shap_analysis", extra=manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
