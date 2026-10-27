#!/usr/bin/env python3
"""
AI-Based Banking Prediction and Fraud Detection Using Dataset and Weight Optimization
=====================================================================================

Two SEPARATE pipelines (their features / targets differ):
  1. Bank Marketing        -> will the customer subscribe to a term deposit?
  2. Credit Card Fraud     -> is the transaction fraudulent?

Every number written to the report is computed by this script from the data you
provide. Ratings are split into three clearly labelled kinds:
  * MEASURED   : metrics from actual train/test experiments and cross-validation
  * RULE-DERIVED: 1-10 ratings computed by transparent formulas from measured
                  data statistics / measured model results (see rate_dataset)
  * JUDGEMENT  : ratings a human must decide (problem relevance, real-world
                  impact). Defaults are editable via --ratings-json.

Usage
-----
  python banking_ml_project.py                      # downloads both datasets with kagglehub
  python banking_ml_project.py --bank bank-full.csv --fraud creditcard.csv
  python banking_ml_project.py --bank bank-full.csv --fraud creditcard.csv --out results
  python banking_ml_project.py --bank ... --fraud ... --quick      # fewer trees / folds
"""
import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
BANK, FRAUD = "Bank Marketing", "Credit Card Fraud"

DATASET_WEIGHTS = {
    "Data quality": 20, "Feature usefulness": 20, "Target-label quality": 15,
    "Problem relevance": 20, "Model compatibility": 10, "Preprocessing ease": 10,
    "Real-world impact": 5,
}
MODEL_WEIGHTS = {
    "Predictive performance": 35, "Robustness": 30, "Speed & Efficiency": 20, "General-purpose suitability": 15,
}
# JUDGEMENT ratings (1-10). Not measurable from the data: edit with --ratings-json.
JUDGEMENT = {
    BANK: {"Problem relevance": 8, "Real-world impact": 7},
    FRAUD: {"Problem relevance": 10, "Real-world impact": 10},
}


# --------------------------------------------------------------------------- loading
KAGGLE_BANK = "janiobachmann/bank-marketing-dataset"   # contains bank.csv (target: deposit)
KAGGLE_FRAUD = "mlg-ulb/creditcardfraud"               # contains creditcard.csv (target: Class)


def kaggle_csv(slug, preferred):
    """Download a Kaggle dataset with kagglehub and return the path of its CSV."""
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("kagglehub is not installed: pip install kagglehub") from e
    folder = Path(kagglehub.dataset_download(slug))
    print(f"  Kaggle dataset {slug} -> {folder}")
    csvs = sorted(folder.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV found in {folder}")
    for c in csvs:
        if c.name.lower() == preferred:
            return str(c)
    return str(max(csvs, key=lambda c: c.stat().st_size))

def read_bank(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip('"') for c in df.columns]
    target = next((c for c in ("y", "deposit", "subscribed") if c in df.columns), None)
    if target is None:
        raise ValueError("Bank file needs a target column named y / deposit / subscribed")
    t = df[target].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    df[target] = t
    return df, target


def read_fraud(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    if "Class" not in df.columns:
        raise ValueError("Fraud file needs a 'Class' column (0 = legitimate, 1 = fraud)")
    return df, "Class"


# --------------------------------------------------------------------------- cleaning
def clean(df, target):
    """Fix types/inconsistencies, drop duplicates & rows without a label. Returns (df, log, stats)."""
    log = {"raw_rows": int(len(df)), "raw_columns": int(df.shape[1])}
    df = df.copy()
    feats = [c for c in df.columns if c != target]

    fixed = []
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            nn = df[c].notna().sum()
            if nn > 0 and conv.notna().sum() >= 0.95 * nn:
                df[c] = conv
                fixed.append(c)
    log["columns_converted_to_numeric"] = fixed

    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat_cols:  # inconsistent text: whitespace, case, blank/placeholder tokens
        s = df[c].astype("string").str.strip().str.lower()
        s = s.mask(s.isin(["", "nan", "none", "null", "?"]))
        df[c] = s.astype(object).where(s.notna(), np.nan)

    log["missing_target_rows_dropped"] = int(df[target].isna().sum())
    df = df[df[target].notna()].copy()
    df[target] = df[target].astype(int)

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["duplicate_rows_dropped"] = int(n_before - len(df))

    missing_cells = int(df[feats].isna().sum().sum())
    unknown_cells = int(sum((df[c] == "unknown").sum() for c in cat_cols))
    conflict_rows = int(df.duplicated(subset=feats, keep=False).sum())  # same features, different label
    counts = df[target].value_counts()
    stats = {
        "n_rows": int(len(df)), "n_features": len(feats), "n_categorical": len(cat_cols),
        "missing_rate": missing_cells / max(1, len(df) * len(feats)),
        "unknown_rate": unknown_cells / max(1, len(df) * max(1, len(cat_cols))),
        "dup_rate": log["duplicate_rows_dropped"] / max(1, log["raw_rows"]),
        "missing_target_rate": log["missing_target_rows_dropped"] / max(1, log["raw_rows"]),
        "conflict_rate": conflict_rows / max(1, len(df)),
        "minority_rate": float(counts.min() / len(df)),
        "imbalance_ratio": float(counts.max() / counts.min()),
    }
    log.update({"clean_rows": int(len(df)), "missing_feature_cells": missing_cells,
                "unknown_category_cells": unknown_cells, "conflicting_label_rows": conflict_rows})
    return df, log, stats


# --------------------------------------------------------------------------- EDA
def eda(name, df, target, figdir, slug):
    figdir.mkdir(parents=True, exist_ok=True)
    num = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c != target and c not in num]

    desc = df.describe(include="all").T
    desc.to_csv(figdir.parent / f"{slug}_summary_statistics.csv")

    # class distribution
    vc = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar([str(i) for i in vc.index], vc.values, color=["#4C78A8", "#E45756"][:len(vc)])
    ax.set_yscale("log"); ax.set_title(f"{name}: class distribution (log y)")
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.2%})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / f"{slug}_class_distribution.png", dpi=130); plt.close(fig)

    # numeric histograms
    cols = num[:12]
    if cols:
        r = int(np.ceil(len(cols) / 4))
        fig, axes = plt.subplots(r, 4, figsize=(14, 2.8 * r)); axes = np.atleast_1d(axes).ravel()
        for a, c in zip(axes, cols):
            a.hist(df[c].dropna(), bins=40, color="#4C78A8"); a.set_title(c, fontsize=9)
        for a in axes[len(cols):]:
            a.axis("off")
        fig.suptitle(f"{name}: numeric feature distributions"); fig.tight_layout()
        fig.savefig(figdir / f"{slug}_numeric_histograms.png", dpi=110); plt.close(fig)

    # correlations with target + heatmap
    if num:
        corr_t = df[num].corrwith(df[target]).dropna()
        top = corr_t.abs().sort_values(ascending=False).head(15).index
        fig, ax = plt.subplots(figsize=(6, 4.5))
        corr_t[top].sort_values().plot.barh(ax=ax, color="#F58518")
        ax.set_title(f"{name}: correlation with target (top 15 |r|)")
        fig.tight_layout(); fig.savefig(figdir / f"{slug}_target_correlation.png", dpi=130); plt.close(fig)

        hm_cols = list(corr_t.abs().sort_values(ascending=False).head(30).index) + [target]
        cm_ = df[hm_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(hm_cols))); ax.set_xticklabels(hm_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(hm_cols))); ax.set_yticklabels(hm_cols, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04); ax.set_title(f"{name}: correlation heatmap")
        fig.tight_layout(); fig.savefig(figdir / f"{slug}_correlation_heatmap.png", dpi=130); plt.close(fig)

        tops = list(corr_t.abs().sort_values(ascending=False).head(4).index)
        fig, axes = plt.subplots(1, len(tops), figsize=(4 * len(tops), 3))
        axes = np.atleast_1d(axes)
        for a, c in zip(axes, tops):
            for k, col in [(0, "#4C78A8"), (1, "#E45756")]:
                a.hist(df.loc[df[target] == k, c].dropna(), bins=40, alpha=.6, density=True,
                       color=col, label=f"class {k}")
            a.set_title(c, fontsize=9); a.legend(fontsize=7)
        fig.suptitle(f"{name}: most target-correlated features by class"); fig.tight_layout()
        fig.savefig(figdir / f"{slug}_top_features_by_class.png", dpi=120); plt.close(fig)

    if "Amount" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for k, col in [(0, "#4C78A8"), (1, "#E45756")]:
            ax.hist(np.log1p(df.loc[df[target] == k, "Amount"]), bins=40, alpha=.6,
                    density=True, color=col, label=f"class {k}")
        ax.set_title("log(1 + Amount) by class"); ax.legend()
        fig.tight_layout(); fig.savefig(figdir / f"{slug}_amount_by_class.png", dpi=130); plt.close(fig)

    cats = [c for c in cat if df[c].nunique() <= 15][:6]
    if cats:
        r = int(np.ceil(len(cats) / 3))
        fig, axes = plt.subplots(r, 3, figsize=(14, 3.6 * r)); axes = np.atleast_1d(axes).ravel()
        for a, c in zip(axes, cats):
            rate = df.groupby(c)[target].mean().sort_values()
            a.barh(rate.index.astype(str), rate.values, color="#54A24B")
            a.axvline(df[target].mean(), color="k", ls="--", lw=1)
            a.set_title(f"positive rate by {c}", fontsize=9)
        for a in axes[len(cats):]:
            a.axis("off")
        fig.suptitle(f"{name}: positive-class rate by category (dashed = overall)")
        fig.tight_layout(); fig.savefig(figdir / f"{slug}_categorical_rates.png", dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- modelling
def make_preprocessor(X, scale):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    tf = [("num", Pipeline(steps), num)]
    if cat:
        tf.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(tf)


def candidates(quick):
    ne = 60 if quick else 200
    return {
        "Logistic Regression": (True, LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": (False, RandomForestClassifier(n_estimators=ne, class_weight="balanced_subsample",
                                                        n_jobs=-1, random_state=SEED)),
        "HistGradientBoosting": (False, HistGradientBoostingClassifier(class_weight="balanced",
                                                                       random_state=SEED)),
    }


def run_pipeline(name, slug, df, target, out, quick):
    figdir, mdir = out / "figures", out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    df, log, stats = clean(df, target)
    print(f"  cleaned rows: {log['clean_rows']:,} (dropped {log['duplicate_rows_dropped']:,} duplicates)")
    eda(name, df, target, figdir, slug)

    X, y = df.drop(columns=[target]), df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    log["train_rows"], log["test_rows"] = int(len(Xtr)), int(len(Xte))
    log["train_positive"], log["test_positive"] = int(ytr.sum()), int(yte.sum())

    cv_cap = 30000 if quick else 100000
    if len(Xtr) > cv_cap:
        Xcv, _, ycv, _ = train_test_split(Xtr, ytr, train_size=cv_cap, stratify=ytr, random_state=SEED)
    else:
        Xcv, ycv = Xtr, ytr
    folds = 3 if (quick or len(Xcv) > 50000) else 5
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    log["cv_rows"], log["cv_folds"] = int(len(Xcv)), folds

    rows, probas, cms, fitted = [], {}, {}, {}
    for mname, (scale, est) in candidates(quick).items():
        print(f"  training {mname} ...")
        pipe = Pipeline([("prep", make_preprocessor(X, scale)), ("model", est)])
        t0 = time.perf_counter(); pipe.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = pipe.predict_proba(Xte)[:, 1]; pred_s = time.perf_counter() - t0
        yhat = (p >= 0.5).astype(int)
        
        # Calculate computational efficiency percentage based on total processing time relative to max time
        cv = cross_validate(pipe, Xcv, ycv, cv=skf, scoring={"f1": "f1", "pr_auc": "average_precision"},
                            n_jobs=1)
        
        rows.append({
            "Model": mname,
            "Accuracy": accuracy_score(yte, yhat),
            "Precision": precision_score(yte, yhat, zero_division=0),
            "Recall": recall_score(yte, yhat, zero_division=0),
            "F1": f1_score(yte, yhat, zero_division=0),
            "ROC-AUC": roc_auc_score(yte, p),
            "PR-AUC": average_precision_score(yte, p),
            "CV F1 mean": cv["test_f1"].mean(), "CV F1 std": cv["test_f1"].std(),
            "CV PR-AUC mean": cv["test_pr_auc"].mean(),
            "Fit s": fit_s, "Predict s": pred_s,
        })
        probas[mname], cms[mname], fitted[mname] = p, confusion_matrix(yte, yhat), pipe
        joblib.dump(pipe, mdir / f"{slug}_{mname.replace(' ', '_')}.joblib")
        
    res = pd.DataFrame(rows).set_index("Model")
    
    # Compute Efficiency % per model: inverse ratio of total time relative to max execution time, scaled to 100%
    total_time = res["Fit s"] + res["Predict s"]
    max_t, min_t = total_time.max(), total_time.min()
    if max_t == min_t:
        res["Efficiency %"] = 100.0
    else:
        # Efficiency percentage where the fastest model gets 100% and others scale downwards proportionally
        res["Efficiency %"] = (min_t / total_time) * 100.0

    # ---- feature importance (Random Forest) -------------------------------------
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
    imp.head(25).to_csv(out / f"{slug}_rf_impurity_importance.csv", header=["importance"])
    pos = yte[yte == 1].index[:1000]; neg = yte[yte == 0].index[:4000]
    sidx = pos.append(neg)
    pi = permutation_importance(rf, Xte.loc[sidx], yte.loc[sidx], scoring="roc_auc",
                                n_repeats=3, random_state=SEED, n_jobs=1)
    perm = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    perm.head(25).to_csv(out / f"{slug}_rf_permutation_importance.csv", header=["roc_auc_drop"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    imp.head(15)[::-1].plot.barh(ax=axes[0], color="#4C78A8"); axes[0].set_title("RF impurity importance (top 15)")
    perm.head(15)[::-1].plot.barh(ax=axes[1], color="#F58518"); axes[1].set_title("RF permutation importance (ROC-AUC drop)")
    fig.suptitle(name); fig.tight_layout(); fig.savefig(figdir / f"{slug}_feature_importance.png", dpi=130); plt.close(fig)

    # ---- charts ---------------------------------------------------------------
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ax = res[metrics].plot.bar(figsize=(9, 4), rot=15, width=.8)
    ax.set_ylim(0, 1.05); ax.set_title(f"{name}: test-set metrics (threshold 0.5)"); ax.legend(ncol=6, fontsize=7)
    plt.tight_layout(); plt.savefig(figdir / f"{slug}_model_comparison.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, p in probas.items():
        fpr, tpr, _ = roc_curve(yte, p); axes[0].plot(fpr, tpr, label=f"{m} ({res.loc[m, 'ROC-AUC']:.3f})")
        pr, rc, _ = precision_recall_curve(yte, p); axes[1].plot(rc, pr, label=f"{m} ({res.loc[m, 'PR-AUC']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
    axes[1].axhline(yte.mean(), color="k", ls="--", lw=.8); axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    for a in axes: a.legend(fontsize=7)
    fig.suptitle(name); fig.tight_layout(); fig.savefig(figdir / f"{slug}_roc_pr_curves.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.8))
    for a, (m, cmx) in zip(np.atleast_1d(axes), cms.items()):
        ConfusionMatrixDisplay(cmx, display_labels=["Negative", "Positive"]).plot(ax=a, colorbar=False, values_format="d")
        a.set_title(m, fontsize=9)
    fig.suptitle(f"{name}: confusion matrices (test set)"); fig.tight_layout()
    fig.savefig(figdir / f"{slug}_confusion_matrices.png", dpi=130); plt.close(fig)

    res.to_csv(out / f"{slug}_model_metrics.csv")
    return {"name": name, "slug": slug, "log": log, "stats": stats, "results": res, "cms": cms,
            "importance": perm, "target": target}


# --------------------------------------------------------------------------- dataset scoring
def auc_to_rating(auc):
    return float(np.clip(1 + 9 * (auc - 0.5) / 0.5, 1, 10))


def rate_dataset(r, judgement):
    s, res = r["stats"], r["results"]
    clip = lambda v: float(np.clip(v, 1, 10))
    ratings = {
        "Data quality": clip(10 - min(9, 50 * s["missing_rate"] + 100 * s["dup_rate"] + 10 * s["unknown_rate"])),
        "Feature usefulness": auc_to_rating(res["ROC-AUC"].max()),
        "Target-label quality": clip(10 - min(9, 100 * s["missing_target_rate"] + 200 * s["conflict_rate"]
                                              + np.log10(s["imbalance_ratio"]))),
        "Model compatibility": auc_to_rating(res["ROC-AUC"].mean()),
        "Preprocessing ease": clip(10 - min(9, 0.3 * s["n_categorical"] + 20 * s["missing_rate"] + 20 * s["dup_rate"]
                                            + 20 * s["unknown_rate"] + 0.5 * np.log10(s["imbalance_ratio"]))),
    }
    ratings.update({k: float(v) for k, v in judgement.items()})
    return {k: ratings[k] for k in DATASET_WEIGHTS}


RATING_KIND = {"Data quality": "rule-derived", "Feature usefulness": "rule-derived (best model ROC-AUC)",
               "Target-label quality": "rule-derived", "Problem relevance": "JUDGEMENT",
               "Model compatibility": "rule-derived (mean model ROC-AUC)", "Preprocessing ease": "rule-derived",
               "Real-world impact": "JUDGEMENT"}


def weighted(ratings, weights):
    return sum(ratings[k] / 10 * w for k, w in weights.items())


# --------------------------------------------------------------------------- model scoring
def score_models(runs):
    per_ds, perf = {}, {}
    for r in runs:
        res = r["results"]
        perf[r["name"]] = res[["F1", "PR-AUC", "ROC-AUC"]].mean(axis=1)
    gp = 10 * pd.concat(perf.values(), axis=1).min(axis=1)
    for r in runs:
        res = r["results"]
        tab = pd.DataFrame({
            "Predictive performance": 10 * perf[r["name"]],
            "Robustness": 10 * (1 - np.minimum(1, (res["CV F1 std"] + (res["CV F1 mean"] - res["F1"]).abs()) / 0.2)),
            "Speed & Efficiency": res["Efficiency %"] / 10.0,  # mapped from 0-100% to a 0-10 scale
            "General-purpose suitability": gp,
        })
        tab["Weighted score /100"] = [weighted(row, MODEL_WEIGHTS) for _, row in tab.iterrows()]
        per_ds[r["name"]] = tab
    overall = sum(per_ds.values()) / len(per_ds)
    return per_ds, overall


# --------------------------------------------------------------------------- report helpers
def md(df, nd=3):
    d = df.reset_index()
    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, row in d.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
                                      for v in row) + " |")
    return "\n".join(out)


def write_report(out, runs, ds_scores, ds_ratings, model_ds, model_overall, args):
    best_ds = max(ds_scores, key=ds_scores.get)
    best_model = model_overall["Weighted score /100"].idxmax()
    L = []
    L += ["# AI-Based Banking Prediction and Fraud Detection Using Dataset and Weight Optimization", "",
          f"*Generated by banking_ml_project.py - scikit-learn {sklearn.__version__}, "
          f"pandas {pd.__version__}, Python {platform.python_version()}. Random seed {SEED}.*", "",
          "## 1. How to read this report", "",
          "- **MEASURED**: computed from your data by train/test (80:20, stratified) experiments and cross-validation.",
          "- **RULE-DERIVED**: 1-10 ratings produced by fixed formulas from measured statistics (Section 4).",
          "- **JUDGEMENT**: ratings no script can measure (problem relevance, real-world impact). "
          "Defaults are editable with `--ratings-json`; they are *not* experimental results.", ""]

    L += ["## 2. Data cleaning (MEASURED)", ""]
    for r in runs:
        L += [f"**{r['name']}**", "", "```json", json.dumps(r["log"], indent=2), "```", ""]

    L += ["## 3. Model results on the held-out 20% test set (MEASURED, decision threshold 0.5)", ""]
    for r in runs:
        cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "CV F1 mean", "CV F1 std", "Fit s", "Predict s", "Efficiency %"]
        L += [f"### {r['name']}", "", md(r["results"][cols]), "",
              f"Test set: {r['log']['test_rows']:,} rows, {r['log']['test_positive']:,} positives "
              f"({r['stats']['minority_rate']:.3%} minority share overall). "
              f"CV: {r['log']['cv_folds']}-fold on {r['log']['cv_rows']:,} training rows.", "",
              f"![models](figures/{r['slug']}_model_comparison.png)", ""]
        cm_rows = []
        for m, c in r["cms"].items():
            cm_rows.append({"Model": m, "TN": int(c[0, 0]), "FP": int(c[0, 1]), "FN": int(c[1, 0]), "TP": int(c[1, 1])})
        L += ["Confusion-matrix counts:", "", md(pd.DataFrame(cm_rows).set_index("Model"), 0), ""]
        if r["name"] == FRAUD:
            L += ["> Fraud is highly imbalanced, so accuracy is dominated by the majority class. "
                  "Read **Precision, Recall, F1 and PR-AUC** first.", ""]

    L += ["## 4. Dataset evaluation and weight allocation", "",
          "Weighted points = (rating / 10) x weight; total = sum, out of 100. Weights follow the project brief: "
          "data quality and problem relevance carry the most (20% each) because a flawed or irrelevant dataset "
          "undermines everything downstream; feature usefulness is equally important (20%); label quality is 15%; "
          "model compatibility and preprocessing ease are 10% each (they affect effort, not validity); "
          "real-world impact is 5%.", ""]
    rows = []
    for k, w in DATASET_WEIGHTS.items():
        row = {"Criterion": k, "Weight %": w, "Rating type": RATING_KIND[k]}
        for n in ds_ratings:
            row[f"{n} rating"] = ds_ratings[n][k]
            row[f"{n} pts"] = ds_ratings[n][k] / 10 * w
        rows.append(row)
    tot = {"Criterion": "**TOTAL /100**", "Weight %": 100, "Rating type": ""}
    for n in ds_ratings:
        tot[f"{n} rating"], tot[f"{n} pts"] = "", ds_scores[n]
    rows.append(tot)
    L += [md(pd.DataFrame(rows).set_index("Criterion"), 2), "",
          "Rule-derived formulas (heuristics chosen by the script author; change them if you disagree):", "",
          "- Data quality = 10 - min(9, 50 x missing rate + 100 x duplicate rate + 10 x 'unknown' rate)",
          "- Feature usefulness = 1 + 9 x (best model ROC-AUC - 0.5) / 0.5",
          "- Target-label quality = 10 - min(9, 100 x missing-label rate + 200 x conflicting-label rate + log10(imbalance ratio))",
          "- Model compatibility = same mapping applied to the *mean* ROC-AUC over all candidate models",
          "- Preprocessing ease = 10 - min(9, 0.3 x #categorical cols + 20 x (missing + duplicate + 'unknown' rates) + 0.5 x log10(imbalance ratio))", "",
          f"**Highest-scoring dataset under this rubric: {best_ds} ({ds_scores[best_ds]:.2f}/100).** "
          "The two datasets pose different tasks, so this ranks them for the rubric's criteria only; it does not "
          "say that one model's metrics are 'better' than the other's.", "",
          "![dataset scores](figures/dataset_scores.png)", ""]

    L += ["## 5. Model evaluation and weight allocation", "",
          "Weights: predictive performance 35% (what we ultimately want), robustness 30% (a model that is good on "
          "one split but unstable is risky), speed & efficiency 20%, general-purpose suitability 15%. Every component is a "
          "1-10 rating computed from MEASURED values:", "",
          "- Predictive performance = 10 x mean(F1, PR-AUC, ROC-AUC) on the test set (minority class)",
          "- Robustness = 10 x (1 - min(1, (CV-F1 std + |CV-F1 mean - test F1|) / 0.2))",
          "- Speed & Efficiency = Normalized computational efficiency percentage (where the fastest pipeline equals 100%) mapped onto a 1-10 scale",
          "- General-purpose suitability = 10 x the model's *worst* performance score across the two datasets", ""]
    for n, t in model_ds.items():
        L += [f"### {n}", "", md(t, 2), ""]
    L += ["### Overall (mean of the two datasets)", "", md(model_overall, 2), "",
          f"**Best model under this rubric: {best_model} ({model_overall.loc[best_model, 'Weighted score /100']:.2f}/100).**", "",
          "![model scores](figures/model_scores.png)", ""]

    L += ["## 6. Charts", ""]
    for r in runs:
        s = r["slug"]
        L += [f"### {r['name']}", ""] + [f"![{t}](figures/{s}_{f}.png)" for t, f in [
            ("class distribution", "class_distribution"), ("numeric histograms", "numeric_histograms"),
            ("correlation heatmap", "correlation_heatmap"), ("feature importance", "feature_importance"),
            ("ROC and PR", "roc_pr_curves"), ("confusion matrices", "confusion_matrices")]] + [""]

    L += ["## 7. Important features (Random Forest permutation importance, top 8, MEASURED)", ""]
    for r in runs:
        L += [f"**{r['name']}**", "", md(r["importance"].head(8).rename("ROC-AUC drop").to_frame(), 4), ""]

    L += ["## 8. Limitations", "",
          "- Ratings for *Problem relevance* and *Real-world impact* are subjective; rule-derived ratings use heuristic "
          "formulas. Different reasonable formulas could change the ranking.",
          "- Test-set ROC-AUC feeds two dataset ratings, so dataset scoring is not independent of the model results.",
          "- Models use default-ish hyperparameters and no tuning; results are a baseline, not a ceiling.",
          "- Decision threshold fixed at 0.5. For fraud, the right operating point depends on the relative cost of "
          "missed fraud vs. false alarms, which this data does not contain.",
          "- Class imbalance is handled with class weights only (no resampling / cost-sensitive threshold search).",
          "- Single random 80:20 split; cross-validation is on a capped training subsample for large data.",
          "- Speed uses single wall-clock timings and varies with hardware and load.",
          "- Bank Marketing: the `duration` feature is only known after the call and leaks outcome information; "
          + ("it was **dropped** in this run." if args.drop_duration else
             "it was **kept** in this run (use `--drop-duration` for a deployable model).") + " "
          "Data is from a single Portuguese bank campaign, so generalisation to other markets is unverified.",
          "- Credit Card Fraud: features V1-V28 are anonymised PCA components (no domain interpretation), "
          "covering two days of European cardholder transactions; fraud patterns drift over time.", "",
          "## 9. Future improvements", "",
          "- Hyper-parameter search (nested CV), calibration, and cost-based threshold selection.",
          "- Resampling comparisons (SMOTE / undersampling), anomaly-detection baselines for fraud.",
          "- Time-based splits for fraud (train on earlier, test on later transactions) to check drift.",
          "- Fairness and explainability review (SHAP) for marketing targeting decisions.",
          "- Replace judgement ratings with a stakeholder-scored panel average; report rating uncertainty.", ""]
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def score_charts(out, ds_ratings, ds_scores, model_overall):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    left = np.zeros(len(ds_ratings))
    for k, w in DATASET_WEIGHTS.items():
        vals = np.array([ds_ratings[n][k] / 10 * w for n in ds_ratings])
        ax.barh(list(ds_ratings), vals, left=left, label=k); left += vals
    for i, n in enumerate(ds_ratings):
        ax.text(left[i] + .5, i, f"{ds_scores[n]:.1f}", va="center")
    ax.set_xlim(0, 110); ax.set_xlabel("weighted points"); ax.set_title("Dataset scores (/100)")
    ax.legend(fontsize=6, ncol=2, loc="lower right"); fig.tight_layout()
    fig.savefig(out / "figures" / "dataset_scores.png", dpi=130); plt.close(fig)

    comps = list(MODEL_WEIGHTS)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(model_overall))
    for k in comps:
        vals = (model_overall[k] / 10 * MODEL_WEIGHTS[k]).values
        ax.barh(model_overall.index, vals, left=left, label=k); left += vals
    for i, v in enumerate(left):
        ax.text(v + .5, i, f"{v:.1f}", va="center")
    ax.set_xlim(0, 110); ax.set_title("Model scores (overall, /100)"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "figures" / "model_scores.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", help="Bank Marketing CSV; if omitted, downloaded via kagglehub")
    ap.add_argument("--fraud", help="Credit Card Fraud CSV; if omitted, downloaded via kagglehub")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="fewer trees / folds / CV rows (faster, noisier)")
    ap.add_argument("--drop-duration", action="store_true", help="drop leaky 'duration' column from bank data")
    ap.add_argument("--ratings-json", help='override judgement ratings, e.g. {"Bank Marketing": {"Problem relevance": 9}}')
    args = ap.parse_args()

    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    judgement = {k: dict(v) for k, v in JUDGEMENT.items()}
    if args.ratings_json:
        for ds, d in json.loads(Path(args.ratings_json).read_text() if Path(args.ratings_json).exists()
                                else args.ratings_json).items():
            judgement[ds].update(d)
    for ds, d in judgement.items():
        for k, v in d.items():
            if not 1 <= v <= 10:
                raise ValueError(f"{ds}/{k}: rating must be 1-10")

    if not args.bank:
        args.bank = kaggle_csv(KAGGLE_BANK, "bank.csv")
    if not args.fraud:
        args.fraud = kaggle_csv(KAGGLE_FRAUD, "creditcard.csv")
    bank_df, bank_t = read_bank(args.bank)
    if args.drop_duration and "duration" in bank_df.columns:
        bank_df = bank_df.drop(columns="duration")
    fraud_df, fraud_t = read_fraud(args.fraud)

    runs = [run_pipeline(BANK, "bank", bank_df, bank_t, out, args.quick),
            run_pipeline(FRAUD, "fraud", fraud_df, fraud_t, out, args.quick)]

    ds_ratings = {r["name"]: rate_dataset(r, judgement[r["name"]]) for r in runs}
    ds_scores = {n: round(weighted(rt, DATASET_WEIGHTS), 2) for n, rt in ds_ratings.items()}
    pd.DataFrame(ds_ratings).assign(Weight=pd.Series(DATASET_WEIGHTS)).to_csv(out / "dataset_ratings.csv")
    pd.Series(ds_scores, name="score_out_of_100").to_csv(out / "dataset_scores.csv")

    model_ds, model_overall = score_models(runs)
    model_overall.to_csv(out / "model_scores_overall.csv")
    for n, t in model_ds.items():
        t.to_csv(out / f"model_scores_{'bank' if n == BANK else 'fraud'}.csv")
    (out / "cleaning_log.json").write_text(json.dumps({r["name"]: {"log": r["log"], "stats": r["stats"]}
                                                       for r in runs}, indent=2))
    score_charts(out, ds_ratings, ds_scores, model_overall)
    write_report(out, runs, ds_scores, ds_ratings, model_ds, model_overall, args)

    print("\nDataset scores:", ds_scores)
    print(model_overall.round(2).to_string())
    print(f"\nDone. Open {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
