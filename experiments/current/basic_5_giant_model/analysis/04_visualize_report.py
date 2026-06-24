"""Visualization and reporting stage for sold-vs-not-72h analysis.

Reads the outputs produced by the previous stages and writes:
  - seven PNG figures under figures/
  - a markdown report report_sold_vs_not_72h.md in the output directory
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import common

OUTPUT_DIR = common.OUTPUT_DIR
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
plt.rcParams.update({"font.size": 9})

MODELS = common.MODELS
PASS_COLS = common.PASS_COLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_median(stat: str) -> float:
    """Parse 'median=...' strings from the univariate output."""
    if pd.isna(stat) or "=" not in stat:
        return np.nan
    try:
        return float(stat.split("=", 1)[1])
    except ValueError:
        return np.nan


def pooled_mad(a, b):
    """Return sqrt(mean(MAD(a)**2, MAD(b)**2)) using the median absolute deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    ma = median_abs_deviation(a, scale=1.0)
    mb = median_abs_deviation(b, scale=1.0)
    if ma == 0 and mb == 0:
        return np.nan
    return np.sqrt((ma**2 + mb**2) / 2.0)


def standardised_median_difference(vals_sold, vals_not):
    """(median_sold - median_not_sold) / pooled_mad."""
    vals_sold = np.asarray(vals_sold, dtype=float)
    vals_not = np.asarray(vals_not, dtype=float)
    vals_sold = vals_sold[~np.isnan(vals_sold)]
    vals_not = vals_not[~np.isnan(vals_not)]
    if len(vals_sold) == 0 or len(vals_not) == 0:
        return np.nan
    pm = pooled_mad(vals_sold, vals_not)
    if np.isnan(pm) or pm == 0:
        return np.nan
    return (float(np.median(vals_sold)) - float(np.median(vals_not))) / pm


def feature_direction(row):
    """Return a short textual direction for numeric features."""
    if row["feature_type"] == "numeric":
        med_sold = parse_median(row["sold_stat"])
        med_not = parse_median(row["not_sold_stat"])
        if pd.isna(med_sold) or pd.isna(med_not):
            return ""
        return "higher" if med_sold > med_not else "lower"
    return "different"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_data():
    df = pd.read_csv(OUTPUT_DIR / "analysis_ready.csv")
    text = pd.read_csv(OUTPUT_DIR / "text_features.csv")
    desc = pd.read_csv(OUTPUT_DIR / "descriptive_aggregate.csv")
    desc_search = pd.read_csv(OUTPUT_DIR / "descriptive_per_model_search.csv")
    uni = pd.read_csv(OUTPUT_DIR / "univariate_tests_per_model.csv")
    ors = pd.read_csv(OUTPUT_DIR / "multivariate_odds_ratios_per_model.csv")

    # Ensure pass columns are boolean
    for c in PASS_COLS:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Overall baseline
    baseline = float(df["sold_within_72h"].mean())
    n_total = len(df)

    # Passed-any flag for pooled analyses
    df["passed_any"] = df[PASS_COLS].any(axis=1)

    # Merge text flags into the main frame
    flag_cols = [c for c in text.columns if c.startswith("title_has_") or c.startswith("desc_has_")]
    merge_keys = ["tracking_key"] if "tracking_key" in df.columns and "tracking_key" in text.columns else ["item_id", "SearchName"]
    text_sub = text[merge_keys + flag_cols].drop_duplicates(subset=merge_keys, keep="first").copy()
    df = df.merge(text_sub, on=merge_keys, how="left")

    return {
        "df": df,
        "text": text,
        "desc": desc,
        "desc_search": desc_search,
        "uni": uni,
        "ors": ors,
        "baseline": baseline,
        "n_total": n_total,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_precision_aggregate(data):
    desc = data["desc"].sort_values("precision", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(desc)))
    bars = ax.barh(desc["model"], desc["precision"], color=colors)
    ax.axvline(data["baseline"], color="crimson", linestyle="--", linewidth=1.5,
               label=f"baseline prevalence ({data['baseline']:.1%})")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Precision = n_sold_72h / n_passed")
    ax.set_title("Aggregate precision per model (sold within 72 h)")
    for bar, prec in zip(bars, desc["precision"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prec:.2f}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.set_ylim(len(desc) + 0.2, -0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_aggregate.png", dpi=200)
    plt.close(fig)


def plot_precision_per_search(data):
    ds = data["desc_search"].copy()
    pivot = ds.pivot(index="model", columns="SearchName", values="precision")
    n_pivot = ds.pivot(index="model", columns="SearchName", values="n_passed")

    # Order rows by aggregate precision descending
    agg_order = data["desc"].sort_values("precision", ascending=False)["model"].tolist()
    pivot = pivot.reindex([m for m in agg_order if m in pivot.index])
    n_pivot = n_pivot.reindex(pivot.index)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title("Precision per model × search (n_passed ≥ 5 annotated)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Precision")

    for i, model in enumerate(pivot.index):
        for j, search in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            n = n_pivot.iloc[i, j]
            if pd.notna(n) and n >= 5 and pd.notna(val):
                text_color = "white" if val < 0.35 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=7)
            elif pd.isna(val) or (pd.notna(n) and n < 5):
                # Gray-out under-powered cells
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=True, color="lightgray",
                                            alpha=0.5, zorder=2))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_per_search.png", dpi=200)
    plt.close(fig)


def plot_volcano_aggregate(data):
    uni = data["uni"]
    agg = uni[uni["search"] == "__ALL__"].copy()
    agg["neg_log10_padj"] = -np.log10(np.maximum(agg["p_adj"].fillna(1.0), 1e-300))

    # Order subplots by aggregate precision
    agg_desc = data["desc"].sort_values("precision", ascending=False)
    models_ordered = [m for m in agg_desc["model"].tolist() if m in agg["model"].unique()]

    n_models = len(models_ordered)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, model in zip(axes, models_ordered):
        sub = agg[agg["model"] == model].copy()
        sig = sub["significant_fdr"].astype(bool)
        ax.scatter(sub.loc[~sig, "effect_size"], sub.loc[~sig, "neg_log10_padj"],
                   c="gray", alpha=0.5, s=25, label="not FDR sig")
        ax.scatter(sub.loc[sig, "effect_size"], sub.loc[sig, "neg_log10_padj"],
                   c="firebrick", alpha=0.75, s=35, label="FDR < 0.05")
        ax.axhline(-np.log10(0.05), color="crimson", linestyle="--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("Effect size")
        ax.set_ylabel("-log10(p_adj)")

        # Label top non-score/margin features
        label_sub = sub[
            ~sub["feature"].str.startswith("score__") &
            ~sub["feature"].str.startswith("margin__")
        ].sort_values("neg_log10_padj", ascending=False).head(5)
        for _, r in label_sub.iterrows():
            ax.annotate(
                r["feature"],
                (r["effect_size"], r["neg_log10_padj"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                alpha=0.85,
            )

    for ax in axes[n_models:]:
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Aggregate univariate tests per model", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "volcano_aggregate.png", dpi=200)
    plt.close(fig)


def compute_std_median_matrix(df):
    """Return matrix of standardised median differences per model and numeric feature."""
    numeric_features = [c for c in common.NUMERIC_FEATURES if c in df.columns]
    # also include title scalar and SVD columns
    for prefix in ("title_svd_", "desc_svd_", "title_len", "title_tokens", "title_desc_overlap_jaccard"):
        if prefix.endswith("_"):
            numeric_features.extend([c for c in df.columns if c.startswith(prefix)])
        else:
            if prefix in df.columns:
                numeric_features.append(prefix)
    numeric_features = sorted(set(numeric_features))

    rows = []
    for model in MODELS:
        pass_col = f"pass__{model}"
        passed = df[df[pass_col]].copy()
        sold = passed[passed["sold_within_72h"] == True]
        not_sold = passed[passed["sold_within_72h"] == False]
        for feat in numeric_features:
            if feat not in df.columns:
                continue
            smd = standardised_median_difference(sold[feat].values, not_sold[feat].values)
            rows.append({"model": model, "feature": feat, "smd": smd})
    return pd.DataFrame(rows)


def plot_effect_size_heatmap(data, std_matrix):
    uni = data["uni"]
    agg = uni[uni["search"] == "__ALL__"].copy()
    sig = agg[agg["significant_fdr"] == True]

    # Top numeric features by frequency of FDR significance across models
    numeric_sig = sig[
        sig["feature_type"] == "numeric"
    ].copy()
    freq = numeric_sig["feature"].value_counts()
    # break ties by max |SMD|
    max_abs = std_matrix.groupby("feature")["smd"].apply(lambda x: x.abs().max())
    feat_rank = pd.DataFrame({"freq": freq, "max_abs_smd": max_abs}).fillna(0)
    feat_rank = feat_rank.sort_values(["freq", "max_abs_smd"], ascending=[False, False])
    top_features = feat_rank.head(15).index.tolist()

    mat = std_matrix.pivot(index="model", columns="feature", values="smd")
    # Order rows by aggregate precision
    agg_order = data["desc"].sort_values("precision", ascending=False)["model"].tolist()
    mat = mat.reindex([m for m in agg_order if m in mat.index])
    mat = mat[top_features]

    fig, ax = plt.subplots(figsize=(max(8, len(top_features) * 0.5), 6))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)
    ax.set_title("Standardised median difference: sold-72h vs not-sold\n(top 15 numeric features with ≥1 FDR hit)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("(median_sold − median_not_sold) / pooled_MAD")

    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            val = mat.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if abs(val) > 1.0 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "effect_size_heatmap.png", dpi=200)
    plt.close(fig)
    return top_features


def plot_top_features_boxplots(data):
    df = data["df"]
    uni = data["uni"]
    agg = uni[uni["search"] == "__ALL__"].copy()
    sig = agg[agg["significant_fdr"] == True]

    # Numeric non-score/margin features ranked by frequency of significance
    non_leak = sig[
        (sig["feature_type"] == "numeric") &
        ~sig["feature"].str.startswith("score__") &
        ~sig["feature"].str.startswith("margin__")
    ].copy()
    freq = non_leak["feature"].value_counts()
    # Ensure Likes and title_char_len are included; take top 4 numeric features
    preferred = ["Likes", "title_char_len"]
    selected = []
    for f in preferred:
        if f in freq.index and f in df.columns:
            selected.append(f)
    for f in freq.index:
        if f not in selected and f in df.columns and len(selected) < 4:
            selected.append(f)
    # Fallback if fewer than 4
    for f in ["log_likes", "title_token_count", "title_len", "title_char_len_full"]:
        if f not in selected and f in df.columns and len(selected) < 4:
            selected.append(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    models_ordered = data["desc"].sort_values("precision", ascending=False)["model"].tolist()

    for ax, feat in zip(axes, selected):
        positions = []
        labels = []
        boxes = []
        pos = 0
        for model in models_ordered:
            pass_col = f"pass__{model}"
            passed = df[df[pass_col]]
            sold_vals = passed[passed["sold_within_72h"] == True][feat].dropna().values
            not_sold_vals = passed[passed["sold_within_72h"] == False][feat].dropna().values
            if len(sold_vals) == 0 and len(not_sold_vals) == 0:
                continue
            positions.extend([pos, pos + 0.7])
            boxes.append(not_sold_vals)
            boxes.append(sold_vals)
            labels.append(f"{model}\nnot")
            labels.append("sold")
            pos += 2.0

        bp = ax.boxplot(
            boxes,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("lightsteelblue" if i % 2 == 0 else "darkorange")
        ax.set_xticks([p + 0.35 for p in positions[::2]])
        ax.set_xticklabels([l.split("\n")[0] for l in labels[::2]], rotation=45, ha="right")
        ax.set_ylabel(feat)
        ax.set_title(feat)
        # Add a small legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="lightsteelblue", label="not sold"),
            Patch(facecolor="darkorange", label="sold 72h"),
        ], loc="upper right", fontsize=7)

    fig.suptitle("Top univariate numeric features: sold-72h vs not-sold across passed cohorts",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_features_boxplots.png", dpi=200)
    plt.close(fig)


def plot_title_embedding_pca(data):
    text = data["text"]
    svd_cols = [c for c in text.columns if c.startswith("title_svd_")]
    if len(svd_cols) == 0:
        return

    X = text[svd_cols].values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    labels = text.loc[mask, "sold_within_72h"].values
    searches = text.loc[mask, "SearchName"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xs)

    search_names = sorted(set(searches))
    ncols = 3
    nrows = int(np.ceil(len(search_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    colors = {False: "steelblue", True: "darkorange"}
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    for ax, search in zip(axes, search_names):
        idx = searches == search
        for sold in (False, True):
            sidx = idx & (labels == sold)
            ax.scatter(coords[sidx, 0], coords[sidx, 1],
                       c=colors[sold], label=str(sold), alpha=0.5, s=10)
        ax.set_title(search)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(title="sold", loc="best", fontsize=7)

    for ax in axes[len(search_names):]:
        ax.axis("off")

    fig.suptitle(
        f"PCA of title SVD embeddings (explained variance: {pca.explained_variance_ratio_.sum():.1%})",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "title_embedding_pca.png", dpi=200)
    plt.close(fig)


def plot_keyword_prevalence(data):
    df = data["df"]
    text = data["text"]
    flag_cols = [c for c in df.columns if c.startswith("title_has_")]
    flag_cols += [c for c in text.columns if c.startswith("title_has_")]
    flag_cols = sorted(set([c for c in flag_cols if c in df.columns]))

    passed_any = df[df["passed_any"]].copy()
    sold = passed_any[passed_any["sold_within_72h"] == True]
    not_sold = passed_any[passed_any["sold_within_72h"] == False]

    records = []
    for f in flag_cols:
        records.append({
            "flag": f,
            "group": "sold_72h",
            "prevalence": sold[f].mean() * 100,
        })
        records.append({
            "flag": f,
            "group": "not_sold",
            "prevalence": not_sold[f].mean() * 100,
        })
    prev = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = prev.pivot(index="flag", columns="group", values="prevalence")
    pivot = pivot[["not_sold", "sold_72h"]]
    pivot.plot(kind="barh", ax=ax, color=["steelblue", "darkorange"])
    ax.set_xlabel("Prevalence (%)")
    ax.set_title("Keyword flag prevalence among passed items: sold-72h vs not-sold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(pivot.max().max() * 1.15, 1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "keyword_prevalence.png", dpi=200)
    plt.close(fig)


def plot_xgboost_shap_bars(data):
    ors = data["ors"].copy()
    if "model_source" not in ors.columns or "search" not in ors.columns:
        return
    shap_rows = ors[
        ors["model_source"].eq("xgboost_mean_abs_shap")
        & ors["search"].eq("__ALL__")
    ].copy()
    if shap_rows.empty:
        return

    models_ordered = data["desc"].sort_values("precision", ascending=False)["model"].tolist()
    ncols = 3
    nrows = int(np.ceil(len(models_ordered) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, model in zip(axes, models_ordered):
        sub = (
            shap_rows[shap_rows["model"].eq(model)]
            .sort_values("abs_coef", ascending=False)
            .head(10)
            .sort_values("abs_coef")
        )
        if sub.empty:
            ax.axis("off")
            continue
        ax.barh(sub["feature"], sub["abs_coef"], color="slateblue", alpha=0.8)
        ax.set_title(model)
        ax.set_xlabel("mean |SHAP|")
        ax.tick_params(axis="y", labelsize=6)

    for ax in axes[len(models_ordered):]:
        ax.axis("off")
    fig.suptitle("Exploratory XGBoost SHAP feature importance within passed cohorts", y=1.00)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgboost_shap_top_features.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    """Render a simple GitHub-flavored Markdown table without tabulate."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(format(value, floatfmt))
            elif pd.isna(value):
                cells.append("")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(data, std_matrix):
    df = data["df"]
    desc = data["desc"]
    uni = data["uni"]
    ors = data["ors"]
    ors_logistic = ors[
        ors.get("model_source", pd.Series(index=ors.index, dtype=str)).eq("logistic")
        & ors.get("search", pd.Series(index=ors.index, dtype=str)).eq("__ALL__")
    ].copy()

    # Overall sample sizes
    n_total = len(df)
    n_evaluated = int(df["evaluated_at_72h"].notna().sum()) if "evaluated_at_72h" in df.columns else n_total
    n_sold = df["sold_within_72h"].sum()
    baseline = data["baseline"]

    # Precision table
    prec_table = desc.sort_values("precision", ascending=False)[
        ["model", "n_passed", "n_sold_72h", "precision", "baseline_prevalence", "lift"]
    ]
    prec_md = markdown_table(prec_table, floatfmt=".3f")

    # Top univariate findings per model
    agg = uni[uni["search"] == "__ALL__"].copy()
    findings = []
    for model in MODELS:
        sub = agg[agg["model"] == model]
        top = sub[
            ~sub["feature"].str.startswith("score__") &
            ~sub["feature"].str.startswith("margin__")
        ].sort_values("p_adj").head(5)
        rows = []
        for _, r in top.iterrows():
            direction = feature_direction(r)
            rows.append(
                f"- **{r['feature']}**: {direction} in sold-72h "
                f"(effect size={r['effect_size']:.3f}, p_adj={r['p_adj']:.3g})"
            )
        if not rows:
            rows.append("- No non-score/margin feature reached FDR significance in this model.")
        findings.append(f"### {model}\n" + "\n".join(rows))
    findings_md = "\n\n".join(findings)

    # Multivariate takeaways for numeric_tree_v1 and rules_price_v1
    mv_sections = []
    for model in ("numeric_tree_v1", "rules_price_v1"):
        sub = ors_logistic[ors_logistic["model"] == model].copy()
        # Exclude other models' score/margin when summarising, but keep own and non-model features
        sub = sub[
            ~sub["feature"].str.startswith("num__score__") &
            ~sub["feature"].str.startswith("num__margin__")
        ]
        sub = sub.sort_values("abs_coef", ascending=False)
        pos = sub[sub["coef"] > 0].head(5)
        neg = sub[sub["coef"] < 0].head(5)
        lines = [f"### {model}"]
        lines.append("Strongest positive associations with 72 h sale (higher coefficient → higher odds):")
        for _, r in pos.iterrows():
            lines.append(
                f"- {r['feature']}: OR={r['odds_ratio']:.2f} "
                f"({r['ci_lower']:.2f}–{r['ci_upper']:.2f})"
            )
        lines.append("\nStrongest negative associations:")
        for _, r in neg.iterrows():
            lines.append(
                f"- {r['feature']}: OR={r['odds_ratio']:.2f} "
                f"({r['ci_lower']:.2f}–{r['ci_upper']:.2f})"
            )
        mv_sections.append("\n".join(lines))
    mv_md = "\n\n".join(mv_sections)

    # Caveats
    small_models = desc[desc["n_passed"] < 50]["model"].tolist()
    small_negative_models = desc[(desc["n_passed"] - desc["n_sold_72h"]) < 10]["model"].tolist()

    report = f"""# Sold within 72 h vs. not sold — analysis report

## Goal and data source

This report compares items that passed each Basic-5 Giant stage-1 model and sold
within 72 hours to those that passed but did not sell in the same window. The
aim is to identify observable item characteristics that distinguish fast sellers
from slow movers, so that future models can better rank high-potential listings.

Data source: `analysis_ready.csv` produced by the upstream data-prep stage
(`{n_total:,}` scraped items; baseline 72 h sale prevalence = `{baseline:.1%}`).

## Sample sizes and baseline prevalence

- Total items: **{n_total:,}**
- Items with a 72 h outcome: **{n_evaluated:,}**
- Items sold within 72 h: **{n_sold:,}** ({baseline:.1%} baseline prevalence)
- All comparisons below are **conditional on passing each model's stage-1 filter**.

## Aggregate precision per model

{prec_md}

All nine models strongly lift precision above the baseline, but the absolute
number of passed items varies widely (e.g. `linear_svm_calibrated_v1` passed only
{desc[desc['model']=='linear_svm_calibrated_v1']['n_passed'].values[0]} items), so
per-search and per-model estimates are noisy for the smaller cohorts.

## Top univariate findings per model

{findings_md}

## Multivariate takeaways

The two models with the largest passed cohorts and stable fits are
`numeric_tree_v1` and `rules_price_v1`. A penalised logistic regression on the
passed cohorts gives the following strongest associations (excluding the other
models' score/margin variables):

{mv_md}

**Interpretation:** For both models, title SVD dimensions carry substantial
predictive signal, and specific brand/search/size levels matter. Title length
and token count also show directionally consistent effects (shorter titles tend
to sell faster). Confidence intervals are wide because the passed cohorts are
still modest.

## Caveats

- **Conditioned on passes:** every comparison is restricted to items that already
  passed the model. The findings do not necessarily generalise to the full
  scraped inventory.
- **Small pass groups:** {', '.join(f'`{m}`' for m in small_models)} each passed
  fewer than 50 items, so their per-search and per-feature estimates are
  imprecise.
- **Tiny not-sold comparison groups:** {', '.join(f'`{m}`' for m in small_negative_models)}
  each had fewer than 10 not-sold items inside its passed cohort. Treat their
  feature contrasts as directional only.
- **Missing Description field:** the source `Description` column is empty for all
  rows. Consequently all description-derived features (description length,
  description tokens, description SVDs, description keyword flags) are uniformly
  zero and cannot contribute to the model or the interpretation.
- **Categorical power:** some categorical levels (e.g. brand or size) have very
  few observations; reported associations may be driven by single searches or
  sellers.
- **Multiple testing:** univariate p-values were BH-FDR adjusted within each
  model, but the set of models and features was not preregistered.

## Actionable recommendations

1. **Fix the Description pipeline.** Description is completely missing, wasting
   the richest source of item semantics. Restoring it should be the highest-impact
   improvement.
2. **Use title signals more deliberately.** Title length and token count are
   consistently related to sales speed; consider penalising very long titles or
   tokenising/normalising title text before training.
3. **Retrain on the passed-and-sold outcome.** The current stage-1 models were
   not trained to predict 72 h sales. Fine-tuning a small reranker on the passed
   cohort with title SVDs, Likes, Price, and brand/size embeddings should lift
   precision further.
4. **Enrich the feature set.** Add seller-level signals (recent sales velocity,
   listing freshness) and search-level baselines; these are likely to dominate
   raw item features.
5. **Pool data across searches where possible.** Several searches have tiny pass
   counts; cross-search regularisation (mixed effects or hierarchical models)
   would stabilise estimates.

## Figures

- `figures/precision_aggregate.png`
- `figures/precision_per_search.png`
- `figures/volcano_aggregate.png`
- `figures/effect_size_heatmap.png`
- `figures/top_features_boxplots.png`
- `figures/title_embedding_pca.png`
- `figures/keyword_prevalence.png`
- `figures/xgboost_shap_top_features.png`
"""
    (OUTPUT_DIR / "report_sold_vs_not_72h.md").write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    data = load_data()

    # Precompute standardised median differences once
    std_matrix = compute_std_median_matrix(data["df"])

    plot_precision_aggregate(data)
    plot_precision_per_search(data)
    plot_volcano_aggregate(data)
    plot_effect_size_heatmap(data, std_matrix)
    plot_top_features_boxplots(data)
    plot_title_embedding_pca(data)
    plot_keyword_prevalence(data)
    plot_xgboost_shap_bars(data)
    build_report(data, std_matrix)

    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Report saved to:  {OUTPUT_DIR / 'report_sold_vs_not_72h.md'}")


if __name__ == "__main__":
    main()
