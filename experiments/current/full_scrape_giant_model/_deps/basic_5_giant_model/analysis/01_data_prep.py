"""Data-prep stage: clean the 72 h frame and build per-model cohorts."""

import sys
from pathlib import Path

# Allow importing the shared config from the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import joblib
import pandas as pd

import common


def main():
    # Load source frame and restrict to rows with a valid 72 h outcome in the
    # six searches named in the analysis request.
    df = common.load_source_frame()
    mask = common.analysis_mask(df)
    df_clean = df.loc[mask].copy()

    # Keep only the columns we need for downstream analysis.
    keep_cols = [c for c in common.KEEP_CORE_COLS if c not in common.RERANKER_LEAK_COLS]
    keep_cols = [c for c in keep_cols if c in df_clean.columns]
    # SearchName appears in both the core list and CATEGORICAL_FEATURES; dedupe.
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    df_clean = df_clean[keep_cols]

    # Save cleaned full frame.
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis_ready_path = common.OUTPUT_DIR / "analysis_ready.csv"
    df_clean.to_csv(analysis_ready_path, index=False)

    # Build per-model passed-item cohorts.
    cohorts = {}
    for model in common.MODELS:
        pass_col = f"pass__{model}"
        cohort = df_clean.loc[df_clean[pass_col].fillna(False)].copy()
        cohorts[model] = cohort

    cohorts_path = common.OUTPUT_DIR / "cohorts.pkl"
    joblib.dump(cohorts, cohorts_path)

    # Compute descriptive statistics per model and per requested search.
    baseline_prevalence = df_clean[common.OUTCOME_COL].mean()

    rows = []
    for model in common.MODELS:
        pass_col = f"pass__{model}"
        cohort = df_clean[df_clean[pass_col].fillna(False)]

        for search_name in common.SEARCHES:
            group = cohort[cohort["SearchName"].eq(search_name)]
            search_scope = df_clean[df_clean["SearchName"].eq(search_name)]
            n_passed = len(group)
            n_sold_72h = int(group[common.OUTCOME_COL].sum())
            search_baseline = search_scope[common.OUTCOME_COL].mean()
            precision = n_sold_72h / n_passed if n_passed > 0 else 0.0
            lift = precision / search_baseline if search_baseline > 0 else float("nan")
            rows.append({
                "model": model,
                "SearchName": search_name,
                "n_evaluated_search": len(search_scope),
                "n_sold_search": int(search_scope[common.OUTCOME_COL].sum()),
                "n_passed": n_passed,
                "n_sold_72h": n_sold_72h,
                "precision": precision,
                "baseline_prevalence": search_baseline,
                "lift": lift,
            })

    desc_per_model_search = pd.DataFrame(rows)
    desc_per_model_search_path = common.OUTPUT_DIR / "descriptive_per_model_search.csv"
    desc_per_model_search.to_csv(desc_per_model_search_path, index=False)

    # Aggregate stats per model across all searches.
    agg_rows = []
    for model in common.MODELS:
        pass_col = f"pass__{model}"
        cohort = df_clean[df_clean[pass_col].fillna(False)]
        n_passed = len(cohort)
        n_sold_72h = int(cohort[common.OUTCOME_COL].sum())
        precision = n_sold_72h / n_passed if n_passed > 0 else 0.0
        lift = precision / baseline_prevalence if baseline_prevalence > 0 else float("nan")
        agg_rows.append({
            "model": model,
            "SearchName": "__ALL__",
            "n_passed": n_passed,
            "n_sold_72h": n_sold_72h,
            "precision": precision,
            "baseline_prevalence": baseline_prevalence,
            "lift": lift,
        })

    desc_aggregate = pd.DataFrame(agg_rows)
    desc_aggregate_path = common.OUTPUT_DIR / "descriptive_aggregate.csv"
    desc_aggregate.to_csv(desc_aggregate_path, index=False)

    # Report summary to stdout.
    print(f"Cleaned frame rows: {len(df_clean):,}")
    print(f"Baseline 72h prevalence: {baseline_prevalence:.4f}")
    print(f"Analysis-ready frame: {analysis_ready_path}")
    print(f"Cohorts dict: {cohorts_path}")
    print(f"Per-model/search stats: {desc_per_model_search_path}")
    print(f"Aggregate stats: {desc_aggregate_path}")


if __name__ == "__main__":
    main()
