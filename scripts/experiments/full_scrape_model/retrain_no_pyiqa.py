"""Retrain full_scrape_plus_visual models without pyiqa features."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
from experiments.full_scrape_model.paths import ensure_experiment_dirs, MODELS_DIR, OFFLINE_RUNS_DIR, run_id, write_json, write_manifest
from experiments.deal_finder import model_sweep as base_sweep

# Best approach per search for full_scrape_plus_visual (from old run)
BEST_APPROACHES = {
    "griffati_donna_all": "numeric_tree_v1",
    "griffati_uomo_all": "numeric_tree_v1",
    "gucci": "logistic_v1_baseline",
    "nike": "numeric_tree_v1",
    "prada": "linear_svm_calibrated_v1",
    "ps4": "linear_svm_calibrated_v1",
}

DATASET_DIR = ROOT / "data" / "experiments" / "full_scrape_model" / "offline_runs" / "sold_status_no_pyiqa_20260518" / "datasets"


def main() -> int:
    ensure_experiment_dirs()
    run_name = run_id("sold_status_no_pyiqa")
    out_dir = OFFLINE_RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    for search_name, approach_name in BEST_APPROACHES.items():
        dataset_path = DATASET_DIR / f"{search_name}.csv"
        if not dataset_path.exists():
            print(f"Skipping {search_name}: dataset not found")
            continue

        df = pd.read_csv(dataset_path, low_memory=False)
        frame = base_sweep.prepare_sweep_frame(df)
        print(f"[{search_name}] rows={len(frame)} eligible={len(frame)}")

        if len(frame) < 50 or frame[base_sweep.TARGET_COL].nunique() < 2:
            print(f"  Skipped: not enough data")
            continue

        spec = next((s for s in base_sweep.APPROACHES if s.name == approach_name), None)
        if spec is None:
            print(f"  Skipped: approach {approach_name} not found")
            continue

        try:
            row = base_sweep.train_approach(
                frame,
                search_name=search_name,
                spec=spec,
                seed=base_sweep.DEFAULT_SEED,
                model_prefix=out_dir.name,
            )
            row["task"] = "sold_status"
            metrics.append(row)
            print(f"  Trained: val_roc={row['validation'].get('roc_auc', 0):.3f} test_roc={row['test'].get('roc_auc', 0):.3f} threshold={row['threshold']:.4f}")
        except Exception as exc:
            print(f"  Error: {exc}")
            import traceback
            traceback.print_exc()

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    # Create best_by_search_mode.csv (only full_scrape_plus_visual mode)
    best_rows = []
    for _, row in metrics_df.iterrows():
        best_rows.append(row.to_dict() | {"feature_mode": "full_scrape_plus_visual"})
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(out_dir / "best_by_search_mode.csv", index=False)

    write_manifest(out_dir / "manifest.json", command="retrain_no_pyiqa", extra={
        "run_dir": str(out_dir),
        "dataset_dir": str(DATASET_DIR),
        "searches": list(BEST_APPROACHES.keys()),
    })
    print(f"\nDone. Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
