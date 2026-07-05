from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.current.time_to_sell._deps.full_scrape_model.paths import (
    EXPERIMENT_ROOT,
    OFFLINE_RUNS_DIR,
    SIMPLE_SCRAPE_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)

TASK_SOLD_STATUS = "sold_status"
TASK_EXTREME_SCORE = "extreme_score"
TASK_CHOICES = (TASK_SOLD_STATUS, TASK_EXTREME_SCORE)

SOLD_STATUS_SOURCE = SIMPLE_SCRAPE_DIR / "full_scrape_merged_stage_resume_20260513" / "backfill_sold_unsold_all_searches.csv"
EXTREME_SCORE_SOURCE = SIMPLE_SCRAPE_DIR / "full_scrape_merged_stage_resume_20260513" / "main_full_scrape_verylow_veryhigh_all_searches.csv"


@dataclass(frozen=True)
class SearchDatasetResult:
    task: str
    search_name: str
    path: Path
    rows: int
    eligible_rows: int
    positive_rows: int
    input_path: str


def normalize_id_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def add_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Dataid" in out.columns:
        out["item_id"] = out["Dataid"].map(normalize_id_value)
    else:
        out["item_id"] = ""
    if "Link" in out.columns:
        link_ids = out["Link"].fillna("").astype(str).str.strip()
        out["item_id"] = out["item_id"].where(out["item_id"].astype(str).str.len() > 0, link_ids)
    return out


def dedupe_by_identity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = add_identity(df)
    out = out[out["item_id"].astype(str).str.len() > 0].copy()
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["item_id"], keep="last")
    return out.reset_index(drop=True)


def read_task_source(task: str) -> pd.DataFrame:
    path = SOLD_STATUS_SOURCE if task == TASK_SOLD_STATUS else EXTREME_SCORE_SOURCE
    if not path.exists():
        raise FileNotFoundError(f"Source dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def available_searches(task: str) -> list[str]:
    df = read_task_source(task)
    if "SourceSearch" not in df.columns:
        return []
    return sorted(df["SourceSearch"].dropna().astype(str).unique().tolist())


def build_search_dataset(task: str, search_name: str) -> pd.DataFrame:
    source = read_task_source(task)
    work = source.loc[source.get("SourceSearch", pd.Series(index=source.index, dtype=object)).astype(str) == str(search_name)].copy()
    if work.empty:
        return work
    work = dedupe_by_identity(work)
    if task == TASK_SOLD_STATUS:
        label = work.get("MergedStatusBinary", pd.Series(index=work.index, dtype=object)).astype(str).str.strip().str.lower()
        work["offline_sold_label"] = label.eq("sold").astype(int)
        work["offline_label_eligible"] = True
        work["label_quality"] = "exact"
        work["label_source"] = "full_scrape_stage_resume_20260513"
    elif task == TASK_EXTREME_SCORE:
        group = work.get("ExtremeScoreGroup", pd.Series(index=work.index, dtype=object)).astype(str).str.strip().str.lower()
        work["offline_sold_label"] = group.eq("score_high").astype(int)
        work["offline_label_eligible"] = group.isin(["score_low", "score_high"])
        work["label_quality"] = "exact"
        work["label_source"] = "main_full_scrape_score_extremes"
    else:
        raise ValueError(f"Unsupported task: {task}")
    work["training_task"] = task
    work["SearchName"] = search_name
    return work.reset_index(drop=True)


def build_datasets(
    *,
    task: str,
    searches: list[str] | None = None,
    out_dir: Path | None = None,
    limit_rows: int | None = None,
) -> list[SearchDatasetResult]:
    ensure_experiment_dirs()
    if out_dir is None:
        out_dir = OFFLINE_RUNS_DIR / run_id(f"{task}_dataset")
    out_dir = assert_experiment_path(out_dir)
    datasets_dir = out_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(searches or [])
    results: list[SearchDatasetResult] = []
    input_path = SOLD_STATUS_SOURCE if task == TASK_SOLD_STATUS else EXTREME_SCORE_SOURCE
    for search_name in available_searches(task):
        if wanted and search_name not in wanted:
            continue
        df = build_search_dataset(task, search_name)
        if limit_rows is not None and limit_rows > 0:
            df = df.head(limit_rows).copy()
        path = datasets_dir / f"{search_name}.csv"
        df.to_csv(path, index=False)
        results.append(
            SearchDatasetResult(
                task=task,
                search_name=search_name,
                path=path,
                rows=int(len(df)),
                eligible_rows=int(df["offline_label_eligible"].sum()) if "offline_label_eligible" in df else 0,
                positive_rows=int(df["offline_sold_label"].sum()) if "offline_sold_label" in df else 0,
                input_path=str(input_path),
            )
        )

    summary = [
        {
            "task": result.task,
            "search_name": result.search_name,
            "path": str(result.path),
            "rows": result.rows,
            "eligible_rows": result.eligible_rows,
            "positive_rows": result.positive_rows,
            "input_path": result.input_path,
        }
        for result in results
    ]
    write_json(out_dir / "dataset_summary.json", {"datasets": summary})
    write_manifest(
        out_dir / "manifest.json",
        command="full_scrape_model.build_dataset",
        extra={"task": task, "dataset_count": len(results), "datasets": summary},
    )
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build per-search full-scrape-model datasets.")
    ap.add_argument("--task", choices=TASK_CHOICES, required=True)
    ap.add_argument("--all-searches", action="store_true")
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")
    out_dir = Path(args.out_dir) if args.out_dir else None
    results = build_datasets(
        task=args.task,
        searches=None if args.all_searches else args.search,
        out_dir=out_dir,
        limit_rows=args.limit_rows,
    )
    for result in results:
        print(
            f"{result.search_name}: rows={result.rows} eligible={result.eligible_rows} "
            f"positive={result.positive_rows} path={result.path}"
        )
    print(f"Output root: {out_dir or EXPERIMENT_ROOT}")


if __name__ == "__main__":
    main()
