"""Inferential-statistics stage: univariate and multivariate associations."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import common

OUTPUT_DIR = common.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
N_PERM_CATEGORICAL = 500
FDR_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

def build_feature_lists(merged_df):
    """Return numeric, categorical, binary feature lists from merged frame."""
    text_numeric_scalar = [
        "title_len",
        "title_tokens",
        "title_desc_overlap_jaccard",
        "title_norm_char_len",
        "title_norm_token_count",
        "title_unique_token_count",
        "title_duplicate_token_ratio",
        "title_avg_token_len",
        "title_punct_count",
        "title_upper_ratio",
        "title_char_len_full_norm",
        "title_token_count_full_norm",
        "title_unique_token_count_full",
        "title_digit_token_count_full",
        "title_keyword_positive_count_full",
        "title_keyword_caution_count_full",
    ]
    title_svd_cols = [c for c in merged_df.columns if c.startswith("title_svd_")]
    desc_svd_cols = [c for c in merged_df.columns if c.startswith("desc_svd_")]
    numeric_features = (
        common.NUMERIC_FEATURES
        + text_numeric_scalar
        + title_svd_cols
        + desc_svd_cols
    )
    numeric_features = [
        c for c in numeric_features
        if c in merged_df.columns
        and merged_df[c].notna().sum() > 0
        and merged_df[c].nunique(dropna=True) > 1
    ]

    categorical_features = [
        "SearchName",
        "Brand",
        "Condition",
        "LocationCountry",
        "Size",
    ]
    categorical_features = [
        c for c in categorical_features
        if c in merged_df.columns
        and merged_df[c].notna().sum() > 0
        and merged_df[c].nunique(dropna=True) > 1
    ]

    text_binary_flags = [
        "title_has_mai_indossato_usato",
        "title_has_new_word_full",
        "title_has_auth_word_full",
        "title_has_limited_word_full",
        "title_has_bundle_word_full",
        "title_has_defect_word_full",
        "title_has_price_like_number_full",
    ]
    binary_features = common.BINARY_FEATURES + text_binary_flags
    binary_features = [
        c for c in binary_features
        if c in merged_df.columns
        and merged_df[c].notna().sum() > 0
        and merged_df[c].nunique(dropna=True) > 1
    ]

    return numeric_features, categorical_features, binary_features


# ---------------------------------------------------------------------------
# Helpers: bootstrap, effect sizes, categorical tests
# ---------------------------------------------------------------------------

def precompute_bootstrap_indices(n_sold, n_not_sold, n_boot=N_BOOTSTRAP, rng=None):
    """Return (sold_idx, not_sold_idx) arrays of shape (n_boot, n_*)."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    sold_idx = rng.integers(0, n_sold, size=(n_boot, n_sold))
    not_idx = rng.integers(0, n_not_sold, size=(n_boot, n_not_sold))
    return sold_idx, not_idx


def bootstrap_median_diff(sold_vals, not_sold_vals, sold_idx=None, not_idx=None):
    """Stratified bootstrap CI for median(sold) - median(not_sold).

    If precomputed index arrays are provided they are used; otherwise a fresh
    set is generated.
    """
    sold_vals = np.asarray(sold_vals, dtype=float)
    not_sold_vals = np.asarray(not_sold_vals, dtype=float)
    n1, n2 = len(sold_vals), len(not_sold_vals)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    if (
        sold_idx is None
        or not_idx is None
        or sold_idx.shape[1] != n1
        or not_idx.shape[1] != n2
    ):
        sold_idx, not_idx = precompute_bootstrap_indices(n1, n2)
    boot_med_sold = np.median(sold_vals[sold_idx], axis=1)
    boot_med_not = np.median(not_sold_vals[not_idx], axis=1)
    diffs = boot_med_sold - boot_med_not
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mannwhitney_effect_size(sold_vals, not_sold_vals):
    """Return (U, pvalue, rank-biserial r)."""
    sold_vals = np.asarray(sold_vals, dtype=float)
    not_sold_vals = np.asarray(not_sold_vals, dtype=float)
    # Remove NaNs for this test
    sold_vals = sold_vals[~np.isnan(sold_vals)]
    not_sold_vals = not_sold_vals[~np.isnan(not_sold_vals)]
    n1, n2 = len(sold_vals), len(not_sold_vals)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    u, p = mannwhitneyu(sold_vals, not_sold_vals, alternative="two-sided")
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u - mu) / sigma if sigma > 0 else 0.0
    r = z / np.sqrt(n1 + n2)
    return float(u), float(p), float(r)


def cramers_v(chi2, n, shape):
    """Compute Cramer's V from chi2 statistic, total n, and table shape."""
    if n == 0 or min(shape) <= 1:
        return np.nan
    return np.sqrt(chi2 / (n * (min(shape) - 1)))


def odds_ratio_2x2_ci(table):
    """Haldane-Anscombe corrected OR and 95% CI for a 2x2 DataFrame.

    table rows: feature=0, feature=1; cols: not-sold, sold.
    """
    a, b = table.iloc[0, 0], table.iloc[0, 1]
    c, d = table.iloc[1, 0], table.iloc[1, 1]
    # Correction avoids zeros
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a_ * d_) / (b_ * c_)
    log_or = np.log(or_val)
    se = np.sqrt(1.0 / a_ + 1.0 / b_ + 1.0 / c_ + 1.0 / d_)
    ci_low = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)
    return float(or_val), float(ci_low), float(ci_high)


def permutation_chi2(table, x, y, n_perm=N_PERM_CATEGORICAL, rng=None):
    """Permutation p-value for chi-square on a contingency table."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    chi2_obs, _, _, _ = chi2_contingency(table)
    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    extremes = 1
    for _ in range(n_perm):
        y_shuf = rng.permutation(y_arr)
        t_perm = pd.crosstab(x_arr, y_shuf)
        if t_perm.shape == table.shape:
            c2, _, _, _ = chi2_contingency(t_perm)
            if c2 >= chi2_obs:
                extremes += 1
    p = extremes / (n_perm + 1)
    return float(chi2_obs), float(p)


def test_categorical_feature(series, outcome, rng=None):
    """Run an appropriate categorical test and return summary dict."""
    table = pd.crosstab(series, outcome)
    n = int(table.sum().sum())
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "p_value": np.nan,
            "effect_size": np.nan,
            "odds_ratio": np.nan,
            "or_ci_lower": np.nan,
            "or_ci_upper": np.nan,
            "contrast_category": "",
            "sold_stat": "",
            "not_sold_stat": "",
            "diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "note": "single category; no test",
        }

    if table.shape == (2, 2):
        chi2, p_chi2, dof, expected = chi2_contingency(table)
        if (expected >= 5).all():
            p = float(p_chi2)
            effect_size = cramers_v(chi2, n, table.shape)
            note = "chi2_2x2"
        else:
            _, p = fisher_exact(table)
            effect_size = cramers_v(chi2, n, table.shape)
            note = "fisher_exact"
        or_val, or_low, or_high = odds_ratio_2x2_ci(table)
        # Proportions of feature=1 in each outcome group
        prop_sold = table.iloc[1, 1] / table.iloc[:, 1].sum()
        prop_not_sold = table.iloc[1, 0] / table.iloc[:, 0].sum()
        contrast = table.index[1]
        return {
            "p_value": float(p),
            "effect_size": float(effect_size),
            "odds_ratio": or_val,
            "or_ci_lower": or_low,
            "or_ci_upper": or_high,
            "contrast_category": str(contrast),
            "sold_stat": f"prop_{contrast}={prop_sold:.3f}",
            "not_sold_stat": f"prop_{contrast}={prop_not_sold:.3f}",
            "diff": float(prop_sold - prop_not_sold),
            "ci_lower": or_low,
            "ci_upper": or_high,
            "note": note,
        }

    # R x 2 table
    chi2, p_chi2, dof, expected = chi2_contingency(table)
    if (expected >= 5).all():
        p = float(p_chi2)
        effect_size = cramers_v(chi2, n, table.shape)
        note = "chi2_contingency"
    else:
        _, p = permutation_chi2(table, series, outcome, rng=rng)
        effect_size = cramers_v(chi2, n, table.shape)
        note = "permutation_chi2"

    # Most informative contrast: category with largest absolute prop difference.
    # Compute proportion of each category within each outcome group.
    group_props = table / table.sum(axis=0)
    diff_series = group_props[True] - group_props[False]
    contrast = diff_series.abs().idxmax()
    prop_sold = group_props.loc[contrast, True]
    prop_not_sold = group_props.loc[contrast, False]

    return {
        "p_value": float(p),
        "effect_size": float(effect_size),
        "odds_ratio": np.nan,
        "or_ci_lower": np.nan,
        "or_ci_upper": np.nan,
        "contrast_category": str(contrast),
        "sold_stat": f"prop_{contrast}={prop_sold:.3f}",
        "not_sold_stat": f"prop_{contrast}={prop_not_sold:.3f}",
        "diff": float(prop_sold - prop_not_sold),
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "note": note,
    }


def benjamini_hochberg(pvals):
    """Return adjusted p-values using the Benjamini-Hochberg procedure."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(m)
    # BH raw
    raw = ranked * m / (np.arange(m) + 1)
    raw = np.minimum(raw, 1.0)
    # Enforce monotonicity from largest to smallest
    adjusted[m - 1] = raw[m - 1]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(raw[i], adjusted[i + 1])
    # Map back
    out = np.empty(m)
    out[order] = adjusted
    return out


# ---------------------------------------------------------------------------
# Univariate analysis
# ---------------------------------------------------------------------------

def run_univariate(merged_df, numeric_features, categorical_features, binary_features):
    """Run all univariate tests and return a DataFrame."""
    rng = np.random.default_rng(RANDOM_STATE)
    searches = ["__ALL__"] + sorted(merged_df["SearchName"].dropna().unique().tolist())
    all_rows = []

    for model in common.MODELS:
        pass_col = f"pass__{model}"
        if pass_col not in merged_df.columns:
            continue
        print(f"  univariate: model={model}")

        for search in searches:
            if search == "__ALL__":
                subset = merged_df.loc[merged_df[pass_col].fillna(False)].copy()
            else:
                subset = merged_df.loc[
                    merged_df[pass_col].fillna(False) & (merged_df["SearchName"] == search)
                ].copy()

            n_passed = len(subset)
            if n_passed < 10:
                all_rows.append({
                    "model": model,
                    "search": search,
                    "feature": "__ALL_FEATURES__",
                    "feature_type": "__SKIP__",
                    "n_passed": n_passed,
                    "n_sold": int(subset[common.OUTCOME_COL].sum()),
                    "n_not_sold": n_passed - int(subset[common.OUTCOME_COL].sum()),
                    "sold_stat": "",
                    "not_sold_stat": "",
                    "diff": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "effect_size": np.nan,
                    "odds_ratio": np.nan,
                    "or_ci_lower": np.nan,
                    "or_ci_upper": np.nan,
                    "p_value": np.nan,
                    "p_adj": np.nan,
                    "significant_fdr": False,
                    "contrast_category": "",
                    "note": f"n_passed={n_passed} < 10",
                    "skipped": True,
                })
                continue

            sold_mask = subset[common.OUTCOME_COL].fillna(False).astype(bool)
            n_sold = int(sold_mask.sum())
            n_not_sold = n_passed - n_sold
            if n_sold < 5 or n_not_sold < 5:
                all_rows.append({
                    "model": model,
                    "search": search,
                    "feature": "__ALL_FEATURES__",
                    "feature_type": "__SKIP__",
                    "n_passed": n_passed,
                    "n_sold": n_sold,
                    "n_not_sold": n_not_sold,
                    "sold_stat": "",
                    "not_sold_stat": "",
                    "diff": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "effect_size": np.nan,
                    "odds_ratio": np.nan,
                    "or_ci_lower": np.nan,
                    "or_ci_upper": np.nan,
                    "p_value": np.nan,
                    "p_adj": np.nan,
                    "significant_fdr": False,
                    "contrast_category": "",
                    "note": f"group sizes {n_sold}/{n_not_sold} < 5",
                    "skipped": True,
                })
                continue

            test_rows = []

            # Precompute bootstrap indices once for this cohort
            sold_idx, not_idx = precompute_bootstrap_indices(n_sold, n_not_sold, rng=rng)

            # Numeric features
            for feat in numeric_features:
                vals = subset[feat]
                if vals.notna().sum() < 5:
                    continue
                sold_vals = vals[sold_mask].dropna().to_numpy(dtype=float)
                not_sold_vals = vals[~sold_mask].dropna().to_numpy(dtype=float)
                if len(sold_vals) < 3 or len(not_sold_vals) < 3:
                    continue
                if np.unique(vals.dropna()).size <= 1:
                    continue

                u, p, r = mannwhitney_effect_size(sold_vals, not_sold_vals)
                med_sold = float(np.median(sold_vals))
                med_not_sold = float(np.median(not_sold_vals))
                ci_low, ci_high = bootstrap_median_diff(
                    sold_vals, not_sold_vals, sold_idx=sold_idx, not_idx=not_idx
                )

                test_rows.append({
                    "model": model,
                    "search": search,
                    "feature": feat,
                    "feature_type": "numeric",
                    "n_passed": n_passed,
                    "n_sold": n_sold,
                    "n_not_sold": n_not_sold,
                    "sold_stat": f"median={med_sold:.4g}",
                    "not_sold_stat": f"median={med_not_sold:.4g}",
                    "diff": float(med_sold - med_not_sold),
                    "ci_lower": float(ci_low),
                    "ci_upper": float(ci_high),
                    "effect_size": float(r),
                    "odds_ratio": np.nan,
                    "or_ci_lower": np.nan,
                    "or_ci_upper": np.nan,
                    "p_value": float(p),
                    "p_adj": np.nan,
                    "significant_fdr": False,
                    "contrast_category": "",
                    "note": f"mannwhitney_u={u:.1f}",
                    "skipped": False,
                })

            # Categorical and binary features
            cat_bin_features = categorical_features + binary_features
            for feat in cat_bin_features:
                if feat == "SearchName" and search != "__ALL__":
                    # Single category within a search; no variance.
                    continue
                series = subset[feat]
                if series.isna().all():
                    continue
                # Treat NaN as its own level for the test
                series = series.fillna("__MISSING__").astype(str)
                if series.nunique() < 2:
                    continue

                res = test_categorical_feature(series, sold_mask, rng=rng)
                if res["note"].startswith("single category"):
                    continue
                test_rows.append({
                    "model": model,
                    "search": search,
                    "feature": feat,
                    "feature_type": "binary" if feat in binary_features else "categorical",
                    "n_passed": n_passed,
                    "n_sold": n_sold,
                    "n_not_sold": n_not_sold,
                    "sold_stat": res["sold_stat"],
                    "not_sold_stat": res["not_sold_stat"],
                    "diff": res["diff"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "effect_size": res["effect_size"],
                    "odds_ratio": res["odds_ratio"],
                    "or_ci_lower": res["or_ci_lower"],
                    "or_ci_upper": res["or_ci_upper"],
                    "p_value": res["p_value"],
                    "p_adj": np.nan,
                    "significant_fdr": False,
                    "contrast_category": res["contrast_category"],
                    "note": res["note"],
                    "skipped": False,
                })

            # FDR correction across all tested features for this (model, search)
            if test_rows:
                pvals = [r["p_value"] for r in test_rows]
                p_adj = benjamini_hochberg(pvals)
                for r, pa in zip(test_rows, p_adj):
                    r["p_adj"] = float(pa)
                    r["significant_fdr"] = bool(pa < FDR_ALPHA)

            all_rows.extend(test_rows)

    cols = [
        "model", "search", "feature", "feature_type", "n_passed", "n_sold",
        "n_not_sold", "sold_stat", "not_sold_stat", "diff", "ci_lower", "ci_upper",
        "effect_size", "odds_ratio", "or_ci_lower", "or_ci_upper", "p_value",
        "p_adj", "significant_fdr", "contrast_category", "note", "skipped",
    ]
    return pd.DataFrame(all_rows, columns=cols)


# ---------------------------------------------------------------------------
# Multivariate analysis
# ---------------------------------------------------------------------------

def preprocess_for_multivariate(X, numeric_features, categorical_features, binary_features):
    """Return a processed design matrix and feature-name list."""
    # Keep only features that exist in X
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]
    binary_features = [c for c in binary_features if c in X.columns]

    # Top-15 + __OTHER__ for categoricals
    cat_transformers = []
    for col in categorical_features:
        vc = X[col].value_counts(dropna=False)
        top_cats = vc.head(15).index.tolist()
        X[col] = X[col].apply(lambda v: v if pd.notna(v) and v in top_cats else "__OTHER__")
        cat_transformers.append(
            (col, OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), [col])
        )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", "passthrough", binary_features),
        ] + cat_transformers,
        remainder="drop",
    )

    X_proc = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return X_proc, feature_names


def logistic_wald_ci(X, y, C=1.0):
    """Fit L2 logistic regression and return coefficient summary with Wald CIs."""
    model = LogisticRegression(
        max_iter=2000,
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    # Approximate covariance: inverse of (X' W X + (1/C) I)
    pred_prob = model.predict_proba(X)[:, 1]
    W = pred_prob * (1 - pred_prob)
    XtWX = X.T @ (X * W[:, None])
    penalty_matrix = (1.0 / C) * np.eye(X.shape[1])
    cov = np.linalg.pinv(XtWX + penalty_matrix)
    se = np.sqrt(np.diag(cov))
    z = 1.96
    ci_lower = coefs - z * se
    ci_upper = coefs + z * se

    return {
        "coef": coefs,
        "intercept": intercept,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def xgboost_shap_rows(X_proc, y, feature_names, *, model: str, search: str):
    """Fit a small XGBoost model and return mean absolute SHAP rows."""
    try:
        from xgboost import XGBClassifier
        import shap
    except Exception as exc:
        return [{
            "model": model,
            "search": search,
            "feature": "__XGBOOST_SHAP_UNAVAILABLE__",
            "coef": np.nan,
            "odds_ratio": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "abs_coef": np.nan,
            "model_source": f"xgboost_shap_error: {type(exc).__name__}: {exc}",
        }]

    xgb = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb.fit(X_proc, y)

    sample_n = min(500, X_proc.shape[0])
    sample_idx = np.linspace(0, X_proc.shape[0] - 1, sample_n, dtype=int)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_proc[sample_idx])
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)

    return [
        {
            "model": model,
            "search": search,
            "feature": name,
            "coef": np.nan,
            "odds_ratio": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "abs_coef": float(value),
            "model_source": "xgboost_mean_abs_shap",
        }
        for name, value in zip(feature_names, mean_abs)
    ]


def run_multivariate(merged_df, numeric_features, categorical_features, binary_features):
    """Run aggregate and per-search multivariate models per stage-1 model."""
    rows = []
    manifest_skipped = []
    search_scopes = ["__ALL__"] + common.SEARCHES

    for model in common.MODELS:
        pass_col = f"pass__{model}"
        for search in search_scopes:
            if search == "__ALL__":
                subset = merged_df.loc[merged_df[pass_col].fillna(False)].copy()
            else:
                subset = merged_df.loc[
                    merged_df[pass_col].fillna(False)
                    & merged_df["SearchName"].eq(search)
                ].copy()

            n_passed = len(subset)
            n_sold = int(subset[common.OUTCOME_COL].sum())
            n_not_sold = n_passed - n_sold

            if n_passed < 50 or n_sold < 10 or n_not_sold < 10:
                manifest_skipped.append({
                    "model": model,
                    "search": search,
                    "n_passed": n_passed,
                    "n_sold": n_sold,
                    "n_not_sold": n_not_sold,
                    "reason": "n_passed<50 or group<10",
                })
                continue

            X = subset[numeric_features + categorical_features + binary_features].copy()
            y = subset[common.OUTCOME_COL].astype(int).values

            try:
                X_proc, feature_names = preprocess_for_multivariate(
                    X, numeric_features, categorical_features, binary_features
                )
            except Exception as exc:
                manifest_skipped.append({
                    "model": model,
                    "search": search,
                    "n_passed": n_passed,
                    "n_sold": n_sold,
                    "n_not_sold": n_not_sold,
                    "reason": f"preprocessing error: {exc}",
                })
                continue

            # Penalised logistic regression.
            try:
                log_summary = logistic_wald_ci(X_proc, y, C=1.0)
                for name, coef, ci_low, ci_high in zip(
                    feature_names, log_summary["coef"], log_summary["ci_lower"], log_summary["ci_upper"]
                ):
                    rows.append({
                        "model": model,
                        "search": search,
                        "feature": name,
                        "coef": float(coef),
                        "odds_ratio": float(np.exp(coef)),
                        "ci_lower": float(np.exp(ci_low)),
                        "ci_upper": float(np.exp(ci_high)),
                        "abs_coef": float(abs(coef)),
                        "model_source": "logistic",
                    })
            except Exception as exc:
                manifest_skipped.append({
                    "model": model,
                    "search": search,
                    "n_passed": n_passed,
                    "n_sold": n_sold,
                    "n_not_sold": n_not_sold,
                    "reason": f"logistic error: {exc}",
                })
                continue

            if search == "__ALL__":
                try:
                    rows.extend(
                        xgboost_shap_rows(
                            X_proc,
                            y,
                            feature_names,
                            model=model,
                            search=search,
                        )
                    )
                except Exception as exc:
                    rows.append({
                        "model": model,
                        "search": search,
                        "feature": "__XGBOOST_SHAP_FAILED__",
                        "coef": np.nan,
                        "odds_ratio": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "abs_coef": np.nan,
                        "model_source": f"xgboost_shap_error: {type(exc).__name__}: {exc}",
                    })

    return pd.DataFrame(rows), manifest_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading inputs...")
    analysis_ready = pd.read_csv(OUTPUT_DIR / "analysis_ready.csv")
    text_features = pd.read_csv(OUTPUT_DIR / "text_features.csv")

    merge_keys = ["tracking_key"] if "tracking_key" in analysis_ready.columns and "tracking_key" in text_features.columns else ["item_id", "SearchName"]

    # Drop duplicate columns from text_features except merge keys.
    dup_cols = [
        c for c in text_features.columns
        if c in analysis_ready.columns and c not in merge_keys
    ]
    text_features = text_features.drop(columns=dup_cols)
    text_features = text_features.drop_duplicates(subset=merge_keys, keep="first")

    merged = analysis_ready.merge(text_features, on=merge_keys, how="left")

    # Drop leak columns if still present
    leak_cols = [c for c in common.RERANKER_LEAK_COLS if c in merged.columns]
    if leak_cols:
        merged = merged.drop(columns=leak_cols)

    print(f"Merged shape: {merged.shape}")

    numeric_features, categorical_features, binary_features = build_feature_lists(merged)
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Binary features: {len(binary_features)}")

    print("\nRunning univariate tests...")
    uni_df = run_univariate(merged, numeric_features, categorical_features, binary_features)
    print(f"  univariate complete: {len(uni_df)} rows")
    uni_path = OUTPUT_DIR / "univariate_tests_per_model.csv"
    uni_df.to_csv(uni_path, index=False)
    print(f"  saved {uni_path} ({len(uni_df)} rows)")

    print("\nRunning multivariate models...")
    multi_df, multi_skipped = run_multivariate(
        merged, numeric_features, categorical_features, binary_features
    )
    print(f"  multivariate complete: {len(multi_df)} rows")
    multi_path = OUTPUT_DIR / "multivariate_odds_ratios_per_model.csv"
    multi_df.to_csv(multi_path, index=False)
    print(f"  saved {multi_path} ({len(multi_df)} rows)")
    if multi_skipped:
        print(f"  skipped multivariate for {len(multi_skipped)} model(s)")

    # Build manifest
    models_analyzed = sorted(uni_df["model"].unique().tolist())
    uni_skipped = uni_df[uni_df["feature"] == "__ALL_FEATURES__"][["model", "search", "note"]].to_dict(orient="records")

    manifest = {
        "models_analyzed_univariate": models_analyzed,
        "n_univariate_rows": len(uni_df),
        "n_multivariate_rows": len(multi_df),
        "n_searches": int(merged["SearchName"].nunique()),
        "n_total_rows": len(merged),
        "univariate_skipped": uni_skipped,
        "multivariate_skipped": multi_skipped,
    }
    manifest_path = OUTPUT_DIR / "stats_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  saved {manifest_path}")

    # Summary counts
    agg_hits = (
        uni_df[(uni_df["search"] == "__ALL__") & (uni_df["significant_fdr"])]
        .groupby("model")
        .size()
        .to_dict()
    )
    n_tests_run = len(uni_df[uni_df["feature"] != "__ALL_FEATURES__"])
    n_model_search_combos = len(
        uni_df[["model", "search"]].drop_duplicates()
    )

    print("\nSummary:")
    print(f"  (model, search) combos tested: {n_model_search_combos}")
    print(f"  individual feature tests run: {n_tests_run}")
    print(f"  significant FDR hits at aggregate level per model: {agg_hits}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
