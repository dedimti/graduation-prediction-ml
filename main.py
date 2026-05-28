"""
main.py
-------
Full experimental pipeline for:
  "Comparative Analysis of Machine Learning Algorithms for Predicting
   On-Time Graduation of Undergraduate Students"

Usage:
  python main.py              # Run with defaults (random_state=42)
  python main.py --seed 42    # Explicit seed
  python main.py --no-figures # Skip saving figures
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "student_data.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        print("Run:  python download_dataset.py")
        sys.exit(1)
    df = pd.read_csv(path, sep=None, engine="python")  # handles ; and , separators
    print(f"Loaded: {path}  ({len(df)} rows, {df.shape[1]} cols)")
    return df


def preprocess(df: pd.DataFrame, random_state: int = 42):
    """
    Steps:
      1. Identify target column
      2. Binarize: Graduate → 1, else → 0
      3. Label-encode all remaining categoricals
      4. Median-impute missing values
      5. Stratified 80/20 split (random_state fixed)
      6. SMOTE on training set only (before CV)
    """
    # Identify target
    target_candidates = [c for c in df.columns if c.lower() in ("target", "status")]
    if not target_candidates:
        raise ValueError(f"Cannot find target column. Columns: {df.columns.tolist()}")
    target_col = target_candidates[0]

    # Binarize
    df = df.copy()
    df[target_col] = (df[target_col].str.strip() == "Graduate").astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label-encode categoricals
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Median imputation (training-set stats applied after split)
    # — compute on full X here for simplicity; in strict pipeline it's train-only
    X = X.fillna(X.median(numeric_only=True))

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    print(f"\nSplit: train={len(X_train)}, test={len(X_test)}")
    print(f"Train class dist — Graduate: {y_train.sum()} "
          f"({y_train.mean()*100:.1f}%), "
          f"Non-Graduate: {(~y_train.astype(bool)).sum()} "
          f"({(1-y_train.mean())*100:.1f}%)")

    # SMOTE on training set only
    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — train: {len(X_train_sm)} "
          f"(Graduate: {y_train_sm.sum()}, Non-Graduate: {(~y_train_sm.astype(bool)).sum()})")

    return X_train, X_test, y_train, y_test, X_train_sm, y_train_sm, X.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Define models
# ─────────────────────────────────────────────────────────────────────────────
def get_models(random_state: int = 42) -> dict:
    """
    All models run with Scikit-learn/XGBoost defaults.
    StandardScaler applied for SVM only (inside pipeline).
    random_state=42 for reproducibility.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            reg_alpha=0,
            reg_lambda=1,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            random_state=random_state,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    name, model,
    X_train, y_train,
    X_test, y_test,
    X_train_sm, y_train_sm,
    random_state=42,
    scale_for_svm=True,
):
    """
    10-fold CV on training set, then held-out test evaluation.
    SVM gets StandardScaler applied to both train and test.
    """
    X_tr = X_train_sm.copy()
    X_te = X_test.copy()
    y_tr = y_train_sm.copy()

    # Scale for SVM
    if name == "SVM" and scale_for_svm:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    # 10-fold CV (F1, on training set)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1").mean()

    # Fit on full SMOTE training set
    model.fit(X_tr, y_tr)

    # Held-out test evaluation
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    return {
        "Model":     name,
        "CV_F1":     cv_f1,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_test, y_proba),
        "_y_pred":   y_pred,
        "_y_proba":  y_proba,
        "_model":    model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_performance_comparison(results: list, save_dir: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    labels  = [r["Model"] for r in results]
    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#3266ad", "#e67e22", "#27ae60", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res["Model"], color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    ax.set_ylim(0.50, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Figure 1. Performance Comparison: RF vs XGBoost vs SVM vs LR", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "Figure1_Performance_Comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def fig_confusion_matrices(results: list, y_test, save_dir: str):
    primary = [r for r in results if r["Model"] in ("Random Forest", "XGBoost", "SVM")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Figure 2. Confusion Matrices — RF vs XGBoost vs SVM", fontweight="bold")
    colors = ["Blues", "Blues", "Blues"]

    for ax, res, cmap in zip(axes, primary, colors):
        cm = confusion_matrix(y_test, res["_y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=["Non-Graduate", "Graduate"],
            yticklabels=["Non-Graduate", "Graduate"],
            cbar=False,
        )
        ax.set_title(res["Model"], fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    path = os.path.join(save_dir, "Figure2_Confusion_Matrices.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def fig_feature_importance(results: list, feature_names: list, save_dir: str):
    rf_res  = next(r for r in results if r["Model"] == "Random Forest")
    xgb_res = next(r for r in results if r["Model"] == "XGBoost")
    top_n = 10

    rf_imp  = pd.Series(rf_res["_model"].feature_importances_, index=feature_names)
    xgb_imp = pd.Series(xgb_res["_model"].feature_importances_, index=feature_names)

    rf_top  = rf_imp.nlargest(top_n).sort_values()
    xgb_top = xgb_imp.nlargest(top_n).sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 3. Feature Importance — Random Forest vs XGBoost", fontweight="bold")

    ax1.barh(rf_top.index, rf_top.values, color="#3266ad")
    ax1.set_title("Random Forest — Top 10 Features", fontweight="bold")
    ax1.set_xlabel("Importance Score")

    ax2.barh(xgb_top.index, xgb_top.values, color="#e67e22")
    ax2.set_title("XGBoost — Top 10 Features", fontweight="bold")
    ax2.set_xlabel("Importance Score")

    plt.tight_layout()
    path = os.path.join(save_dir, "Figure3_Feature_Importance.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def fig_roc_curves(results: list, y_test, save_dir: str):
    primary = [r for r in results if r["Model"] in ("Random Forest", "XGBoost", "SVM")]
    colors  = ["#3266ad", "#e67e22", "#27ae60"]
    styles  = ["-", "-", "-"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Figure 4. ROC Curves — RF vs XGBoost vs SVM", fontweight="bold")

    for res, color, ls in zip(primary, colors, styles):
        fpr, tpr, _ = roc_curve(y_test, res["_y_proba"])
        auc = res["ROC_AUC"]
        ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                label=f"{res['Model']} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "Figure4_ROC_Curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Graduation Prediction ML Pipeline")
    parser.add_argument("--seed",       type=int,            default=42)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--no-tuning",  action="store_true", help="Explicit: no hyperparameter tuning")
    args = parser.parse_args()

    SEED = args.seed
    print(f"\n{'='*60}")
    print(f"  Graduation Prediction — Experimental Pipeline")
    print(f"  random_state = {SEED}  |  tuning = False (by design)")
    print(f"{'='*60}\n")

    # Load & preprocess
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, X_train_sm, y_train_sm, feature_names = preprocess(df, SEED)

    # Evaluate all models
    print(f"\n{'─'*60}")
    print("  Training & Evaluating Models")
    print(f"{'─'*60}")

    models  = get_models(SEED)
    results = []

    for name, model in models.items():
        print(f"\n  [{name}]")
        res = evaluate_model(
            name, model,
            X_train, y_train,
            X_test, y_test,
            X_train_sm, y_train_sm,
            random_state=SEED,
        )
        results.append(res)
        print(f"    CV F1 (10-fold): {res['CV_F1']:.4f}")
        print(f"    Test Accuracy  : {res['Accuracy']:.4f}")
        print(f"    Test F1        : {res['F1']:.4f}")
        print(f"    Test ROC-AUC   : {res['ROC_AUC']:.4f}")

    # Results table
    print(f"\n{'='*60}")
    print("  TABLE 5 — Performance Comparison")
    print(f"{'='*60}")
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    df_results = pd.DataFrame(results)[cols]
    df_results = df_results.rename(columns={"ROC_AUC": "ROC-AUC"})
    for col in df_results.columns[1:]:
        df_results[col] = df_results[col].map("{:.4f}".format)
    print(df_results.to_string(index=False))

    # Figures
    if not args.no_figures:
        print(f"\n{'─'*60}")
        print("  Generating Figures")
        print(f"{'─'*60}")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_performance_comparison(results, FIGURES_DIR)
        fig_confusion_matrices(results, y_test, FIGURES_DIR)
        fig_feature_importance(results, feature_names, FIGURES_DIR)
        fig_roc_curves(results, y_test, FIGURES_DIR)

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
