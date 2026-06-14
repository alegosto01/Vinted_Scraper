from __future__ import annotations
import pandas as pd
import numpy as np


def _nonempty_str(series: pd.Series) -> pd.Series:
    """Return boolean Series True where series is not NA and non-empty string (case-insensitive)."""
    return series.notna() & (series.astype(str).str.strip().str.lower().isin(["", "nan", "none", "[]"]) == False)


def offline_image_filter(frame: pd.DataFrame, search_col: str = "SearchName") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter offline rows by main-image availability.

    Keeps rows with either VisiblePictureCount > 0 or PrimaryImageUrl present.

    Returns:
        (kept_frame, accounting_df) where accounting_df has columns:
        [search_name, total_rows, kept_rows, skipped_rows, skipped_no_picture,
         skipped_picture_count_missing, skipped_other]
    """
    if len(frame) == 0:
        empty_accounting = pd.DataFrame({
            "search_name": ["__TOTAL__"],
            "total_rows": [0],
            "kept_rows": [0],
            "skipped_rows": [0],
            "skipped_no_picture": [0],
            "skipped_picture_count_missing": [0],
            "skipped_other": [0],
        })
        return frame.copy(), empty_accounting

    # Get search names, defaulting to "__ALL__" if column missing
    search_names = frame.get(search_col, "__ALL__")
    if not isinstance(search_names, pd.Series):
        search_names = pd.Series("__ALL__", index=frame.index)
    elif search_col not in frame.columns:
        search_names = pd.Series("__ALL__", index=frame.index)

    # Parse VisiblePictureCount
    vpc = pd.to_numeric(frame.get("VisiblePictureCount"), errors="coerce")

    # Check for usable PrimaryImageUrl
    has_url = _nonempty_str(frame.get("PrimaryImageUrl", pd.Series(index=frame.index, dtype=object)))

    # Determine usable rows
    usable = (vpc > 0) | has_url

    # Compute skip reasons for skipped rows
    skipped_mask = ~usable
    skip_reasons = np.empty(len(frame), dtype=object)
    skip_reasons[:] = "other"

    # picture_count_missing: vpc is NaN AND not has_url
    skip_reasons[skipped_mask & vpc.isna() & ~has_url] = "picture_count_missing"

    # no_picture: vpc <= 0 (including 0) AND not has_url
    skip_reasons[skipped_mask & (vpc <= 0) & ~has_url] = "no_picture"

    # Build accounting table
    kept = frame[usable].copy()

    search_group = search_names[~usable]
    skip_reason_group = skip_reasons[skipped_mask]

    accounting_rows = []
    for search_val in frame[search_col].unique() if search_col in frame.columns else ["__ALL__"]:
        mask = search_names == search_val
        total = mask.sum()
        kept_count = (mask & usable).sum()
        skipped_count = (mask & skipped_mask).sum()

        skipped_sub = skip_reason_group[search_group == search_val]
        skipped_no_picture = (skipped_sub == "no_picture").sum()
        skipped_picture_count_missing = (skipped_sub == "picture_count_missing").sum()
        skipped_other = (skipped_sub == "other").sum()

        accounting_rows.append({
            "search_name": search_val,
            "total_rows": total,
            "kept_rows": kept_count,
            "skipped_rows": skipped_count,
            "skipped_no_picture": skipped_no_picture,
            "skipped_picture_count_missing": skipped_picture_count_missing,
            "skipped_other": skipped_other,
        })

    accounting_df = pd.DataFrame(accounting_rows)

    # Add __TOTAL__ row
    total_row = {
        "search_name": "__TOTAL__",
        "total_rows": accounting_df["total_rows"].sum(),
        "kept_rows": accounting_df["kept_rows"].sum(),
        "skipped_rows": accounting_df["skipped_rows"].sum(),
        "skipped_no_picture": accounting_df["skipped_no_picture"].sum(),
        "skipped_picture_count_missing": accounting_df["skipped_picture_count_missing"].sum(),
        "skipped_other": accounting_df["skipped_other"].sum(),
    }
    accounting_df = pd.concat([accounting_df, pd.DataFrame([total_row])], ignore_index=True)

    return kept, accounting_df


def live_image_filter(frame: pd.DataFrame, search_col: str = "SearchName") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter live rows by image availability.

    Keeps rows with either Images or collector_visual_features_path present.

    Returns:
        (kept_frame, accounting_df) where accounting_df has columns:
        [search_name, total_rows, kept_rows, skipped_rows, skipped_no_image]
    """
    if len(frame) == 0:
        empty_accounting = pd.DataFrame({
            "search_name": ["__TOTAL__"],
            "total_rows": [0],
            "kept_rows": [0],
            "skipped_rows": [0],
            "skipped_no_image": [0],
        })
        return frame.copy(), empty_accounting

    # Get search names, defaulting to "__ALL__" if column missing
    search_names = frame.get(search_col, "__ALL__")
    if not isinstance(search_names, pd.Series):
        search_names = pd.Series("__ALL__", index=frame.index)
    elif search_col not in frame.columns:
        search_names = pd.Series("__ALL__", index=frame.index)

    # Check for usable Images
    has_images = _nonempty_str(frame.get("Images", pd.Series(index=frame.index, dtype=object)))

    # Check for usable visual features path
    has_visual = _nonempty_str(frame.get("collector_visual_features_path", pd.Series(index=frame.index, dtype=object)))

    # Determine usable rows
    usable = has_images | has_visual

    # Build accounting table
    kept = frame[usable].copy()

    skipped_mask = ~usable
    search_group = search_names[skipped_mask]

    accounting_rows = []
    for search_val in frame[search_col].unique() if search_col in frame.columns else ["__ALL__"]:
        mask = search_names == search_val
        total = mask.sum()
        kept_count = (mask & usable).sum()
        skipped_count = (mask & skipped_mask).sum()
        skipped_no_image = skipped_count  # All skipped are due to "no_image"

        accounting_rows.append({
            "search_name": search_val,
            "total_rows": total,
            "kept_rows": kept_count,
            "skipped_rows": skipped_count,
            "skipped_no_image": skipped_no_image,
        })

    accounting_df = pd.DataFrame(accounting_rows)

    # Add __TOTAL__ row
    total_row = {
        "search_name": "__TOTAL__",
        "total_rows": accounting_df["total_rows"].sum(),
        "kept_rows": accounting_df["kept_rows"].sum(),
        "skipped_rows": accounting_df["skipped_rows"].sum(),
        "skipped_no_image": accounting_df["skipped_no_image"].sum(),
    }
    accounting_df = pd.concat([accounting_df, pd.DataFrame([total_row])], ignore_index=True)

    return kept, accounting_df
