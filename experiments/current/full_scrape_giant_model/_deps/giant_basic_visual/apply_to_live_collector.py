#!/usr/bin/env python3
"""Shadow-only live scorer for main-image Basic5 visual models."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.current.full_scrape_giant_model._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import score_with_model  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.giant_basic_visual.features import (  # noqa: E402
    FEATURE_MODES,
    filter_rows_with_main_image,
    is_diagnostic_mode,
    merge_quality_scores,
    mode_live_ready,
)
from experiments.current.full_scrape_giant_model._deps.giant_basic_visual.paths import (  # noqa: E402
    LIVE_SHADOW_DIR,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
)


SCORE_PREFIX = "visual_score__"
THRESHOLD_PREFIX = "visual_threshold__"
PASS_PREFIX = "visual_pass__"
DEFAULT_LIVE_MODES = ("main_image_simple", "main_image_scores")
MODEL_SEARCHES = (
    "griffati_donna_all",
    "griffati_uomo_all",
    "gucci",
    "nike",
    "prada",
    "ps4",
    "donna_accessori_gioielli",
    "telefoni",
    "hobby_collezionismo",
)
ID_COLUMNS = [
    "SearchName",
    "item_id",
    "Dataid",
    "Title",
    "Brand",
    "Size",
    "Price",
    "Likes",
    "Link",
    "LocalPrimaryImagePath",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def latest_offline_run() -> Path:
    runs = sorted((path for path in OFFLINE_RUNS_DIR.glob("giant_basic_visual_*") if path.is_dir()), key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError("No giant_basic_visual offline runs found. Train the experiment first.")
    return runs[-1]


def selected_modes(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_LIVE_MODES
    unknown = sorted(set(values) - set(FEATURE_MODES))
    if unknown:
        raise ValueError(f"Unknown feature modes: {', '.join(unknown)}")
    return tuple(dict.fromkeys(values))


def load_live_tracked(live_run_dir: Path) -> pd.DataFrame:
    path = live_run_dir / "tracked_items.csv"
    if not path.exists():
        raise FileNotFoundError(f"No tracked_items.csv found in {live_run_dir}")
    return pd.read_csv(path, low_memory=False)


def sanitize_search_name(value: str) -> str:
    import re

    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def search_onehot_column(search: str) -> str:
    return f"search__{sanitize_search_name(search)}"


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SearchName" not in out.columns:
        out["SearchName"] = ""
    out["SearchName"] = out["SearchName"].fillna("").astype(str)

    for col in ["Title", "Brand", "Size"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    for col in ["Price", "Likes"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "Page" not in out.columns:
        out["Page"] = np.nan
    out["Page"] = pd.to_numeric(out["Page"], errors="coerce")

    out = base_sweep.add_engineered_snapshot_features(out)
    search_values = out["SearchName"].fillna("").astype(str)
    for search in MODEL_SEARCHES:
        out[search_onehot_column(search)] = search_values.eq(search).astype(int)
    return out


def load_visual_model_records(
    *,
    offline_run: str | None = None,
    modes: tuple[str, ...] = DEFAULT_LIVE_MODES,
    approaches: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    run_dir = OFFLINE_RUNS_DIR / offline_run if offline_run else latest_offline_run()
    metrics_path = run_dir / "metrics_long.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing visual metrics table: {metrics_path}")
    metrics = pd.read_csv(metrics_path, low_memory=False)
    trained = metrics[metrics["status"].astype(str).eq("trained")].copy()
    trained = trained[trained["feature_mode"].astype(str).isin(modes)].copy()
    if approaches:
        trained = trained[trained["approach"].astype(str).isin(set(approaches))].copy()
    records: list[dict[str, Any]] = []
    for _, row in trained.iterrows():
        feature_mode = str(row["feature_mode"])
        metadata_path = Path(str(row["metadata_path"]))
        artifact_path = Path(str(row["artifact_path"]))
        if not metadata_path.exists():
            metadata_path = MODELS_DIR / metadata_path.name
        if not artifact_path.exists():
            artifact_path = MODELS_DIR / artifact_path.name
        metadata = load_json(metadata_path)
        metadata_live_ready = bool(metadata.get("live_ready", mode_live_ready(feature_mode)))
        diagnostic = bool(metadata.get("diagnostic", is_diagnostic_mode(feature_mode))) or is_diagnostic_mode(feature_mode)
        records.append(
            {
                "feature_mode": feature_mode,
                "approach": str(row["approach"]),
                "artifact_path": artifact_path,
                "metadata_path": metadata_path,
                "metadata": metadata,
                "threshold": float(metadata.get("threshold", row.get("threshold", 1.0))),
                "live_ready": bool(metadata_live_ready and mode_live_ready(feature_mode)),
                "diagnostic": diagnostic,
            }
        )
    if not records:
        raise ValueError("No trained visual model records matched the selected modes/approaches")
    return records


def prepare_shadow_frame(tracked: pd.DataFrame, visual_source: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prepared = prepare_feature_frame(tracked)
    filtered, skipped = filter_rows_with_main_image(prepared)
    filtered, visual_merge = merge_quality_scores(filtered, visual_source)
    return filtered, skipped, visual_merge


def ensure_model_columns(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for col in metadata.get("numeric_features", []):
        if col not in out.columns:
            out[col] = np.nan
    for col in metadata.get("text_features", []):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def apply_visual_models(tracked: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    scored = tracked.copy()
    for record in records:
        key = f"{record['feature_mode']}__{record['approach']}"
        score_col = f"{SCORE_PREFIX}{key}"
        threshold_col = f"{THRESHOLD_PREFIX}{key}"
        pass_col = f"{PASS_PREFIX}{key}"
        model = load_model(record["artifact_path"])
        feature_frame = ensure_model_columns(scored, record["metadata"])
        probs = np.clip(np.asarray(score_with_model(model, feature_frame), dtype=float), 0.0, 1.0)
        threshold = float(record["threshold"])
        scored[score_col] = probs
        scored[threshold_col] = threshold
        scored[pass_col] = scored[score_col] >= threshold
    return scored


def build_shadow_candidates(scored: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, row in scored.iterrows():
        passed: list[str] = []
        best_key = ""
        best_score = np.nan
        best_threshold = np.nan
        best_margin = -np.inf
        for record in records:
            key = f"{record['feature_mode']}__{record['approach']}"
            pass_col = f"{PASS_PREFIX}{key}"
            if not bool(row.get(pass_col, False)):
                continue
            passed.append(key)
            score = pd.to_numeric(pd.Series([row.get(f"{SCORE_PREFIX}{key}")]), errors="coerce").iloc[0]
            threshold = pd.to_numeric(pd.Series([row.get(f"{THRESHOLD_PREFIX}{key}")]), errors="coerce").iloc[0]
            if pd.isna(score) or pd.isna(threshold):
                continue
            margin = float(score - threshold)
            if margin > best_margin:
                best_margin = margin
                best_key = key
                best_score = float(score)
                best_threshold = float(threshold)
        if not passed:
            continue
        out = row.copy()
        out["VisualPassedModels"] = ", ".join(passed)
        out["VisualPassedModelCount"] = int(len(passed))
        out["VisualBestModel"] = best_key
        out["VisualBestScore"] = best_score
        out["VisualBestThreshold"] = best_threshold
        out["VisualBestMargin"] = best_margin if np.isfinite(best_margin) else np.nan
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["VisualPassedModelCount", "VisualBestMargin", "VisualBestScore"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def compare_current_giant_candidates(tracked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from experiments.current.full_scrape_giant_model._deps.basic_5_giant_model import apply_to_live_collector as giant_live

        records = giant_live.load_model_records(tuple(giant_live.BASIC5_APPROACH_NAMES), offline_run=None)
        scored = giant_live.apply_models(tracked, records)
        candidates = giant_live.build_telegram_candidates(scored, records)
        return candidates, {"status": "ok", "candidate_rows": int(len(candidates))}
    except Exception as exc:
        return pd.DataFrame(), {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def write_report(
    path: Path,
    *,
    live_run_dir: Path,
    scored: pd.DataFrame,
    skipped: pd.DataFrame,
    shadow_candidates: pd.DataFrame,
    current_candidates: pd.DataFrame,
    compare_status: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    lines = [
        "# Giant Basic Visual - Live Shadow",
        "",
        f"Live collector: `{live_run_dir}`",
        "",
        "Shadow-only run. No Telegram messages were sent and current Basic5 Telegram policy was not changed.",
        "Only `LocalPrimaryImagePath` was used; rows without a readable main image were skipped.",
        "",
        "## Counts",
        "",
        f"- Rows scored by visual models: `{len(scored)}`",
        f"- Rows skipped by main-image filter: `{len(skipped)}`",
        f"- Visual shadow candidates: `{len(shadow_candidates)}`",
        f"- Current giant Telegram candidates for comparison: `{len(current_candidates)}`",
        f"- Current-giant comparison status: `{json.dumps(compare_status, sort_keys=True)}`",
        "",
        "## Models",
        "",
        "| mode | approach | live ready | diagnostic | threshold | passed |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for record in records:
        key = f"{record['feature_mode']}__{record['approach']}"
        pass_col = f"{PASS_PREFIX}{key}"
        passed = int(scored[pass_col].sum()) if pass_col in scored.columns else 0
        lines.append(
            f"| {record['feature_mode']} | {record['approach']} | {bool(record['live_ready'])} | "
            f"{bool(record['diagnostic'])} | {float(record['threshold']):.4f} | {passed} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    out_dir: Path,
    *,
    live_run_dir: Path,
    scored: pd.DataFrame,
    skipped: pd.DataFrame,
    shadow_candidates: pd.DataFrame,
    current_candidates: pd.DataFrame,
    compare_status: dict[str, Any],
    records: list[dict[str, Any]],
    visual_merge: dict[str, Any],
) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "live_shadow_scored_items.csv"
    skipped_path = out_dir / "skipped_main_image_rows.csv"
    candidates_path = out_dir / "live_shadow_candidates.csv"
    current_path = out_dir / "current_giant_candidates.csv"
    report_path = out_dir / "live_shadow_report.md"

    scored.to_csv(scored_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    shadow_candidates.to_csv(candidates_path, index=False)
    current_candidates.to_csv(current_path, index=False)
    write_report(
        report_path,
        live_run_dir=live_run_dir,
        scored=scored,
        skipped=skipped,
        shadow_candidates=shadow_candidates,
        current_candidates=current_candidates,
        compare_status=compare_status,
        records=records,
    )
    manifest = {
        "mode": "shadow_only",
        "telegram_policy_changed": False,
        "live_run_dir": str(live_run_dir),
        "models": [
            {
                "feature_mode": str(record["feature_mode"]),
                "approach": str(record["approach"]),
                "artifact_path": str(record["artifact_path"]),
                "metadata_path": str(record["metadata_path"]),
                "threshold": float(record["threshold"]),
                "live_ready": bool(record["live_ready"]),
                "diagnostic": bool(record["diagnostic"]),
            }
            for record in records
        ],
        "visual_merge": visual_merge,
        "outputs": {
            "scored": str(scored_path),
            "skipped": str(skipped_path),
            "shadow_candidates": str(candidates_path),
            "current_giant_candidates": str(current_path),
            "report": str(report_path),
        },
        "current_giant_compare": compare_status,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    live_run_dir = Path(args.live_run_dir)
    modes = selected_modes(args.feature_mode)
    approaches = tuple(args.approach) if args.approach else None
    out_dir = Path(args.out_dir) if args.out_dir else LIVE_SHADOW_DIR / run_id("live_shadow")
    visual_source = Path(args.visual_source) if args.visual_source else None

    records = load_visual_model_records(offline_run=args.offline_run, modes=modes, approaches=approaches)
    tracked = load_live_tracked(live_run_dir)
    filtered, skipped, visual_merge = prepare_shadow_frame(tracked, visual_source)
    scored = apply_visual_models(filtered, records)
    shadow_candidates = build_shadow_candidates(scored, records)
    current_candidates, compare_status = compare_current_giant_candidates(tracked)
    manifest = write_outputs(
        out_dir,
        live_run_dir=live_run_dir,
        scored=scored,
        skipped=skipped,
        shadow_candidates=shadow_candidates,
        current_candidates=current_candidates,
        compare_status=compare_status,
        records=records,
        visual_merge=visual_merge,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow-score live collector rows with main-image Basic5 visual models.")
    parser.add_argument("--live-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--offline-run", default=None)
    parser.add_argument("--feature-mode", action="append", choices=FEATURE_MODES)
    parser.add_argument("--approach", action="append")
    parser.add_argument("--visual-source", default="")
    parser.add_argument("--shadow-only", action="store_true", default=True, help="Required behavior: score only, never send.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    if not args.shadow_only:
        raise ValueError("giant_basic_visual live scoring is shadow-only")
    run_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
