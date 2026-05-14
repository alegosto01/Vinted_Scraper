from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip import check_item


BLUR_PROMPT = "this image is blurry"
SHARP_PROMPT = "this image is NOT blurry"


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    mean_diff = float(np.mean(group_a) - np.mean(group_b))
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled_denom = (((len(group_a) - 1) * var_a) + ((len(group_b) - 1) * var_b)) / (len(group_a) + len(group_b) - 2)
    if pooled_denom <= 0:
        return float("nan")
    return mean_diff / math.sqrt(pooled_denom)


def permutation_pvalue(group_a: np.ndarray, group_b: np.ndarray, *, n_perm: int = 2000, seed: int = 0) -> float:
    if len(group_a) == 0 or len(group_b) == 0:
        return float("nan")
    observed = abs(float(np.mean(group_a) - np.mean(group_b)))
    combined = np.concatenate([group_a, group_b]).astype(np.float64)
    n_a = len(group_a)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(int(n_perm)):
        rng.shuffle(combined)
        diff = abs(float(np.mean(combined[:n_a]) - np.mean(combined[n_a:])))
        if diff >= observed:
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def _write_progress(out_dir: Path, payload: dict) -> None:
    (out_dir / "clip_blur_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"processed={payload.get('processed', 0)}",
        f"total={payload.get('total', 0)}",
        f"remaining={payload.get('remaining', 0)}",
        f"n_scored={payload.get('n_scored', 0)}",
        f"n_failures={payload.get('n_failures', 0)}",
        f"resume={payload.get('resume', False)}",
    ]
    (out_dir / "clip_blur_progress.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(input_csv: Path, out_dir: Path, *, batch_size: int = 4, resume: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    required = {"Dataid", "SoldLabel", "LocalPrimaryImagePath"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    scores_path = out_dir / "clip_blur_item_scores.csv"
    failures_path = out_dir / "clip_blur_failures.csv"

    if resume and scores_path.exists():
        try:
            existing_scores = pd.read_csv(scores_path)
        except pd.errors.EmptyDataError:
            existing_scores = pd.DataFrame(columns=["Dataid", "Title", "SoldLabel", "BalancedLabel", "LocalPrimaryImagePath", "ClipBlurProb", "ClipNotBlurProb"])
    else:
        existing_scores = pd.DataFrame(columns=["Dataid", "Title", "SoldLabel", "BalancedLabel", "LocalPrimaryImagePath", "ClipBlurProb", "ClipNotBlurProb"])
    if resume and failures_path.exists():
        try:
            existing_failures = pd.read_csv(failures_path)
        except pd.errors.EmptyDataError:
            existing_failures = pd.DataFrame(columns=["Dataid", "reason", "image_path"])
    else:
        existing_failures = pd.DataFrame(columns=["Dataid", "reason", "image_path"])

    rows = existing_scores.to_dict(orient="records")
    failures = existing_failures.to_dict(orient="records")
    scored_ids = set(existing_scores["Dataid"].dropna().tolist()) if not existing_scores.empty and "Dataid" in existing_scores.columns else set()
    valid = df[df["LocalPrimaryImagePath"].notna()].copy()
    total = len(valid)
    if scored_ids and "Dataid" in valid.columns:
        valid = valid[~valid["Dataid"].isin(scored_ids)].copy()
    batch_rows = []
    batch_paths: list[str] = []
    processed = len(rows)
    next_progress = ((processed // 25) + 1) * 25
    _write_progress(
        out_dir,
        {
            "processed": processed,
            "total": total,
            "remaining": max(0, total - processed),
            "n_scored": len(rows),
            "n_failures": len(failures),
            "resume": resume,
        },
    )
    for row in valid.itertuples(index=False):
        image_path = Path(str(row.LocalPrimaryImagePath))
        if not image_path.exists():
            failures.append({"Dataid": getattr(row, "Dataid", None), "reason": "missing_local_file", "image_path": str(image_path)})
            processed += 1
            _write_progress(
                out_dir,
                {
                    "processed": processed,
                    "total": total,
                    "remaining": max(0, total - processed),
                    "n_scored": len(rows),
                    "n_failures": len(failures),
                    "resume": resume,
                    "last_dataid": getattr(row, "Dataid", None),
                },
            )
            continue
        batch_rows.append(row)
        batch_paths.append(str(image_path))
        if len(batch_rows) < batch_size and processed + len(batch_rows) < total:
            continue
        try:
            batch_probs = check_item(BLUR_PROMPT, SHARP_PROMPT, batch_paths)
        except Exception as exc:
            for failed_row, failed_path in zip(batch_rows, batch_paths):
                failures.append(
                    {
                        "Dataid": getattr(failed_row, "Dataid", None),
                        "reason": f"{type(exc).__name__}: {exc}",
                        "image_path": failed_path,
                    }
                )
            processed += len(batch_rows)
            batch_rows, batch_paths = [], []
            print(f"processed {processed}/{total}", flush=True)
            pd.DataFrame(rows).drop_duplicates(subset=["Dataid"], keep="last").to_csv(scores_path, index=False)
            pd.DataFrame(failures).to_csv(failures_path, index=False)
            _write_progress(
                out_dir,
                {
                    "processed": processed,
                    "total": total,
                    "remaining": max(0, total - processed),
                    "n_scored": len(rows),
                    "n_failures": len(failures),
                    "resume": resume,
                    "last_dataid": getattr(batch_rows[-1], "Dataid", None) if batch_rows else None,
                },
            )
            continue

        for current_row, current_path, probs in zip(batch_rows, batch_paths, np.asarray(batch_probs)):
            blurry_prob, sharp_prob = probs
            rows.append(
                {
                    "Dataid": getattr(current_row, "Dataid", None),
                    "Title": getattr(current_row, "Title", ""),
                    "SoldLabel": int(getattr(current_row, "SoldLabel")),
                    "BalancedLabel": getattr(current_row, "BalancedLabel", ""),
                    "LocalPrimaryImagePath": current_path,
                    "ClipBlurProb": float(blurry_prob),
                    "ClipNotBlurProb": float(sharp_prob),
                }
            )
        processed += len(batch_rows)
        last_dataid = getattr(batch_rows[-1], "Dataid", None) if batch_rows else None
        batch_rows, batch_paths = [], []
        if processed >= next_progress or processed == total:
            print(f"processed {processed}/{total}", flush=True)
            next_progress += 25
        if rows:
            pd.DataFrame(rows).drop_duplicates(subset=["Dataid"], keep="last").to_csv(scores_path, index=False)
        if failures:
            pd.DataFrame(failures).to_csv(failures_path, index=False)
        _write_progress(
            out_dir,
            {
                "processed": processed,
                "total": total,
                "remaining": max(0, total - processed),
                "n_scored": len(rows),
                "n_failures": len(failures),
                "resume": resume,
                "last_dataid": last_dataid,
            },
        )

    result_df = pd.DataFrame(rows).drop_duplicates(subset=["Dataid"], keep="last")
    result_df.to_csv(scores_path, index=False)
    pd.DataFrame(failures).to_csv(failures_path, index=False)

    sold = result_df.loc[result_df["SoldLabel"] == 1, "ClipBlurProb"].to_numpy(dtype=np.float64)
    unsold = result_df.loc[result_df["SoldLabel"] == 0, "ClipBlurProb"].to_numpy(dtype=np.float64)

    summary = {
        "input_csv": str(input_csv),
        "n_scored": int(len(result_df)),
        "n_failures": int(len(failures)),
        "n_sold": int(len(sold)),
        "n_unsold": int(len(unsold)),
        "prompt_positive": BLUR_PROMPT,
        "prompt_negative": SHARP_PROMPT,
        "sold_mean_blur_prob": float(np.mean(sold)) if len(sold) else None,
        "unsold_mean_blur_prob": float(np.mean(unsold)) if len(unsold) else None,
        "sold_median_blur_prob": float(np.median(sold)) if len(sold) else None,
        "unsold_median_blur_prob": float(np.median(unsold)) if len(unsold) else None,
        "sold_blurry_rate_p50": float(np.mean(sold >= 0.5)) if len(sold) else None,
        "unsold_blurry_rate_p50": float(np.mean(unsold >= 0.5)) if len(unsold) else None,
        "mean_diff_sold_minus_unsold": float(np.mean(sold) - np.mean(unsold)) if len(sold) and len(unsold) else None,
        "cohen_d": cohen_d(sold, unsold),
        "permutation_pvalue": permutation_pvalue(sold.copy(), unsold.copy()) if len(sold) and len(unsold) else None,
    }

    with open(out_dir / "clip_blur_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_progress(
        out_dir,
        {
            "processed": total,
            "total": total,
            "remaining": 0,
            "n_scored": len(result_df),
            "n_failures": len(failures),
            "resume": resume,
            "done": True,
        },
    )

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run CLIP blur prompt analysis on a labeled dataset with local image paths.")
    ap.add_argument("--input", required=True, help="CSV with SoldLabel and LocalPrimaryImagePath columns.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--no_resume", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_analysis(
        Path(args.input),
        Path(args.out_dir),
        batch_size=max(1, int(args.batch_size)),
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
