"""Multimodal 'beat giant_basic_visual' experiment (CPU, frozen encoders + light heads).

Tests whether adding frozen title-text embeddings and/or richer heads beats the
champion's tabular+visual GBDT. Encoders are frozen (no GPU fine-tuning): title ->
multilingual MiniLM sentence embedding; visual -> the already-computed dino/aesthetic
columns (incl. raw 384-d dino vector when present). Heads: LogReg, HistGBDT (champion),
MLP, TabNet, FT-Transformer.

Usage:
  python run_experiment.py --csv <data.csv> --label label_sold_within_24h \
      --feature-sets tab,tab+vis,tab+txt,tab+vis+txt --heads logreg,gbdt,mlp,tabnet,ft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VISUAL_PREFIXES = ("Dino", "Aesthetic", "BadPhoto", "MainImage", "Photo", "Pyiqa",
                   "SimpleBadPhotoScore", "CombinedBadPhotoScore", "QualityScore")


# ---------- features ----------
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def build_blocks(df: pd.DataFrame):
    """Return dict of feature-block name -> (DataFrame of numeric cols)."""
    blocks = {}
    # tabular: log price/likes + search one-hots
    tab = pd.DataFrame(index=df.index)
    tab["logPrice"] = np.log1p(_num(df.get("Price")).clip(lower=0))
    tab["logLikes"] = np.log1p(_num(df.get("Likes")).clip(lower=0))
    if "SearchName" in df.columns:
        for s in sorted(df["SearchName"].dropna().unique()):
            tab[f"search__{s}"] = (df["SearchName"] == s).astype(float)
    blocks["tab"] = tab

    # visual: numeric cols matching known visual prefixes (excludes string status cols)
    vis_cols = [c for c in df.columns
                if any(c.startswith(p) or c == p for p in VISUAL_PREFIXES)
                and _num(df[c]).notna().any()]
    blocks["vis"] = df[vis_cols].apply(_num) if vis_cols else pd.DataFrame(index=df.index)
    return blocks


def text_embeddings(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    titles = df.get("Title").fillna("").astype(str).tolist()
    key = hashlib.md5((tag + "|" + "|".join(titles)).encode()).hexdigest()[:16]
    cache = CACHE / f"txt_{key}.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(TEXT_MODEL, device="cpu")
        emb = model.encode(titles, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True)
        np.save(cache, emb)
    return pd.DataFrame(emb, index=df.index,
                        columns=[f"txt_{i}" for i in range(emb.shape[1])])


def load_emb_block(name: str, tag: str, mask: np.ndarray, index) -> pd.DataFrame | None:
    """Load a precomputed embedding block (embed_blocks.py) aligned to filtered rows."""
    p = CACHE / f"{name}_{tag}.npy"
    if not p.exists():
        return None
    arr = np.load(p)[mask]
    return pd.DataFrame(arr, index=index, columns=[f"{name}_{i}" for i in range(arr.shape[1])])


def assemble(blocks, spec: str) -> pd.DataFrame:
    parts = [blocks[t] for t in spec.split("+") if t in blocks and blocks[t] is not None
             and blocks[t].shape[1] > 0]
    return pd.concat(parts, axis=1)


# ---------- metrics ----------
def precision_at_k(y, scores, frac=0.10):
    n = max(1, int(len(y) * frac))
    idx = np.argsort(scores)[::-1][:n]
    return float(np.mean(np.asarray(y)[idx]))


def choose_threshold(y_val, s_val, min_precision=0.40, min_count=10):
    best = None
    for t in np.unique(s_val):
        pred = s_val >= t
        if pred.sum() < min_count:
            continue
        prec = float(np.mean(np.asarray(y_val)[pred]))
        if prec >= min_precision:
            best = t if best is None else min(best, t)
    return best


# ---------- heads ----------
def fit_score(head, Xtr, ytr, Xva, Xte):
    imp = SimpleImputer(strategy="median")
    if head == "gbdt":
        m = HistGradientBoostingClassifier(random_state=SEED, learning_rate=0.05,
                                            max_iter=400, l2_regularization=1.0)
        m.fit(Xtr, ytr)
        f = lambda X: m.predict_proba(X)[:, 1]
        return f(Xva), f(Xte)
    Xtr_i = imp.fit_transform(Xtr)
    Xva_i, Xte_i = imp.transform(Xva), imp.transform(Xte)
    if head == "logreg":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        m.fit(Xtr_i, ytr)
        return m.predict_proba(Xva_i)[:, 1], m.predict_proba(Xte_i)[:, 1]
    if head == "mlp":
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                        early_stopping=True, max_iter=300, random_state=SEED))
        m.fit(Xtr_i, ytr)
        return m.predict_proba(Xva_i)[:, 1], m.predict_proba(Xte_i)[:, 1]
    if head == "tabnet":
        return _tabnet(Xtr_i, ytr, Xva_i, Xte_i)
    if head == "ft":
        return _ft(Xtr_i, ytr, Xva_i, Xte_i)
    raise ValueError(head)


def _tabnet(Xtr, ytr, Xva, Xte):
    from pytorch_tabnet.tab_model import TabNetClassifier
    ss = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = ss.transform(Xtr), ss.transform(Xva), ss.transform(Xte)
    m = TabNetClassifier(seed=SEED, verbose=0, n_d=16, n_a=16, n_steps=3)
    m.fit(Xtr.astype(np.float32), ytr.astype(int),
          eval_set=[(Xtr.astype(np.float32), ytr.astype(int))],
          max_epochs=120, patience=20, batch_size=256, virtual_batch_size=128)
    return m.predict_proba(Xva.astype(np.float32))[:, 1], m.predict_proba(Xte.astype(np.float32))[:, 1]


def _ft(Xtr, ytr, Xva, Xte):
    """Compact FT-Transformer: per-feature linear tokenizer + [CLS] + encoder."""
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    ss = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(ss.transform(Xtr), dtype=torch.float32)
    Xva_t = torch.tensor(ss.transform(Xva), dtype=torch.float32)
    Xte_t = torch.tensor(ss.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    F = Xtr_t.shape[1]
    d = 32

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(F, d) * 0.02)
            self.b = nn.Parameter(torch.zeros(F, d))
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, 4, d * 2, dropout=0.3, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, 2)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.w + self.b            # (B,F,d)
            z = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
            z = self.tr(z)
            return self.head(z[:, 0]).squeeze(-1)

    net = FT()
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(1, ytr.sum())], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    yva = np.asarray(ytr)  # unused; val AUC below uses Xva/actual val labels passed separately
    best_state, best_auc, bad = None, -1, 0
    n = Xtr_t.shape[0]
    for epoch in range(120):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            out = net(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = torch.sigmoid(net(Xva_t)).numpy()
        # early stop on train-vs-noise proxy: use val ranking auc vs pseudo? need val labels
        auc = _FT_VAL_AUC[0](va) if _FT_VAL_AUC else 0
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xva_t)).numpy(), torch.sigmoid(net(Xte_t)).numpy()


_FT_VAL_AUC = []  # set per-run so FT can early-stop on real val AUC


# ---------- driver ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", default="label_sold_within_24h")
    ap.add_argument("--feature-sets", default="tab,tab+vis,tab+txt,tab+vis+txt")
    ap.add_argument("--heads", default="logreg,gbdt,mlp,tabnet,ft")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--emb-tag", default=None, help="tag of precomputed embed_blocks npy files")
    args = ap.parse_args()

    df_full = pd.read_csv(args.csv, low_memory=False)
    mask_pos = _num(df_full[args.label]).notna().to_numpy()
    df = df_full[mask_pos].reset_index(drop=True)
    y = _num(df[args.label]).astype(int).to_numpy()
    tag = args.tag or Path(args.csv).parent.name
    etag = args.emb_tag or tag
    print(f"data={args.csv}\nlabel={args.label} rows={len(df)} pos_rate={y.mean():.3f}")

    blocks = build_blocks(df)
    if any("txt" in fs.split("+") for fs in args.feature_sets.split(",")):
        print("encoding titles (multilingual MiniLM)...")
        blocks["txt"] = text_embeddings(df, tag)
    # precomputed encoder blocks (embed_blocks.py), aligned via the label mask
    for name, key in [("img_clip", "img"), ("txt_clip", "ctxt"),
                      ("txt_bert", "btxt"), ("txt_mdeberta", "dtxt")]:
        b = load_emb_block(name, etag, mask_pos, df.index)
        if b is not None:
            blocks[key] = b
    print("blocks: " + " ".join(f"{k}={v.shape[1]}d" for k, v in blocks.items() if v is not None and v.shape[1]))

    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=y)
    va, te = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=y[rest])
    print(f"split train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    rows = []
    for fs in args.feature_sets.split(","):
        X = assemble(blocks, fs).to_numpy(dtype=float)
        for head in args.heads.split(","):
            if head == "ft" and X.shape[1] > 80:
                # FT-Transformer tokenizes each feature -> O(F^2) attention; impractical
                # on CPU for the 400-800-dim embedding sets (and wrong to tokenize raw
                # embedding dims). Restrict FT to the compact tabular feature set.
                print(f"  {fs:12s} {head:7s} SKIP (n_feat={X.shape[1]}>80, CPU-infeasible)")
                continue
            _FT_VAL_AUC.clear()
            _FT_VAL_AUC.append(lambda s: roc_auc_score(y[va], s))
            try:
                sv, stst = fit_score(head, X[tr], y[tr], X[va], X[te])
            except Exception as exc:  # noqa: BLE001
                print(f"  {fs:12s} {head:7s} FAILED: {exc}")
                continue
            thr = choose_threshold(y[va], sv)
            if thr is not None:
                pred = stst >= thr
                p = float(np.mean(y[te][pred])) if pred.sum() else float("nan")
                r = float(np.mean(pred[y[te] == 1])) if (y[te] == 1).sum() else float("nan")
            else:
                p = r = float("nan")
            row = {
                "feature_set": fs, "head": head, "n_feat": X.shape[1],
                "roc_auc": round(roc_auc_score(y[te], stst), 4),
                "pr_auc": round(average_precision_score(y[te], stst), 4),
                "P@10%": round(precision_at_k(y[te], stst, 0.10), 4),
                "prec@thr": round(p, 4) if p == p else None,
                "rec@thr": round(r, 4) if r == r else None,
            }
            rows.append(row)
            print(f"  {fs:12s} {head:7s} AUC={row['roc_auc']:.3f} PR={row['pr_auc']:.3f} "
                  f"P@10%={row['P@10%']:.3f} prec@thr={row['prec@thr']}")

    res = pd.DataFrame(rows).sort_values(["roc_auc"], ascending=False)
    out = CACHE / f"results_{tag}_{args.label}.csv"
    res.to_csv(out, index=False)
    print(f"\n=== RESULTS ({tag}, {args.label}) sorted by ROC-AUC ===")
    print(res.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
