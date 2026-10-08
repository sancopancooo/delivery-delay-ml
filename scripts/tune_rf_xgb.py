"""Random Forest + XGBoost hiperparametre aramasi (RandomizedSearchCV)."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_ready_v1.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Masaustune de kopyala (kullanici oradan takip ediyor)
DESK_OUT = Path.home() / "Desktop" / "delivery-delay-ml" / "outputs"

CAT_COLS = [
    "Shipping Mode",
    "Category Name",
    "Order Region",
    "Customer Segment",
]
TARGET = "Late_delivery_risk"


def score_on_test(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision_gecikme": round(float(precision_score(y_test, pred)), 4),
        "recall_gecikme": round(float(recall_score(y_test, pred)), 4),
        "f1_gecikme": round(float(f1_score(y_test, pred)), 4),
    }


def save_both(path: Path, data: bytes | str, text: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if text:
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    if DESK_OUT.parent.exists():
        desk_path = DESK_OUT / path.name
        desk_path.parent.mkdir(parents=True, exist_ok=True)
        if text:
            desk_path.write_text(data, encoding="utf-8")
        else:
            desk_path.write_bytes(data)


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET].astype(int)
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=CAT_COLS, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    results = {}

    print(">>> RF RandomizedSearch basliyor...")
    t0 = perf_counter()
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    rf_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, 20, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    }
    rf_search = RandomizedSearchCV(
        rf,
        param_distributions=rf_space,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_train, y_train)
    rf_test = score_on_test(rf_search.best_estimator_, X_test, y_test)
    results["random_forest"] = {
        "best_params": rf_search.best_params_,
        "best_cv_auc": round(float(rf_search.best_score_), 4),
        "test": rf_test,
        "seconds": round(perf_counter() - t0, 1),
    }
    print("RF best:", rf_search.best_params_, "CV AUC", results["random_forest"]["best_cv_auc"])
    print("RF test:", rf_test)

    print(">>> XGB RandomizedSearch basliyor...")
    t1 = perf_counter()
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=neg / pos,
        random_state=42,
        n_jobs=-1,
    )
    xgb_space = {
        "n_estimators": [150, 250, 350, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb,
        param_distributions=xgb_space,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_test = score_on_test(xgb_search.best_estimator_, X_test, y_test)
    results["xgboost"] = {
        "best_params": xgb_search.best_params_,
        "best_cv_auc": round(float(xgb_search.best_score_), 4),
        "test": xgb_test,
        "seconds": round(perf_counter() - t1, 1),
    }
    print("XGB best:", xgb_search.best_params_, "CV AUC", results["xgboost"]["best_cv_auc"])
    print("XGB test:", xgb_test)

    json_text = json.dumps(results, indent=2, ensure_ascii=False)
    (OUT / "hyperparam_rf_xgb.json").write_text(json_text, encoding="utf-8")
    if DESK_OUT.parent.exists():
        DESK_OUT.mkdir(parents=True, exist_ok=True)
        (DESK_OUT / "hyperparam_rf_xgb.json").write_text(json_text, encoding="utf-8")

    cmp = pd.DataFrame(
        [
            {
                "model": "Logistic Regression (baseline)",
                "accuracy": 0.6940,
                "roc_auc": 0.7239,
                "precision_gecikme": 0.8460,
                "recall_gecikme": 0.5410,
                "f1_gecikme": 0.6600,
            },
            {
                "model": "Decision Tree (tuned)",
                "accuracy": 0.6968,
                "roc_auc": 0.7335,
                "precision_gecikme": 0.8485,
                "recall_gecikme": 0.5442,
                "f1_gecikme": 0.6631,
            },
            {
                "model": "Random Forest (onceki)",
                "accuracy": 0.6967,
                "roc_auc": 0.7325,
                "precision_gecikme": 0.8497,
                "recall_gecikme": 0.5429,
                "f1_gecikme": 0.6625,
            },
            {"model": "Random Forest (tuned)", **rf_test},
            {
                "model": "XGBoost (onceki)",
                "accuracy": 0.6973,
                "roc_auc": 0.7305,
                "precision_gecikme": 0.8468,
                "recall_gecikme": 0.5468,
                "f1_gecikme": 0.6645,
            },
            {"model": "XGBoost (tuned)", **xgb_test},
        ]
    )
    cmp.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if DESK_OUT.parent.exists():
        cmp.to_csv(DESK_OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [("RF", 0.7325, rf_test["roc_auc"]), ("XGB", 0.7305, xgb_test["roc_auc"])]
    ax = axes[0]
    x = np.arange(len(pairs))
    ax.bar(x - 0.2, [p[1] for p in pairs], 0.4, label="Onceki")
    ax.bar(x + 0.2, [p[2] for p in pairs], 0.4, label="Tuned")
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylim(0.70, 0.76)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Hiperparametre: onceki vs tuned")
    ax.legend(fontsize=8)
    for i, (_, a, b) in enumerate(pairs):
        ax.text(i - 0.2, a + 0.001, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.001, f"{b:.4f}", ha="center", fontsize=8)

    ax2 = axes[1]
    ax2.barh(cmp["model"], cmp["roc_auc"], color="#4C78A8")
    ax2.set_xlim(0.70, 0.76)
    ax2.set_xlabel("ROC-AUC (test)")
    ax2.set_title("Tum modeller")
    for i, v in enumerate(cmp["roc_auc"]):
        ax2.text(v + 0.0008, i, f"{v:.4f}", va="center", fontsize=8)
    fig.suptitle("Hiperparametre Optimizasyonu — RF & XGBoost", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "hyperparam_comparison.png", dpi=150, bbox_inches="tight")
    if DESK_OUT.parent.exists():
        fig.savefig(DESK_OUT / "hyperparam_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig2, axc = plt.subplots(figsize=(11.5, 5.2))
    metric_cols = ["accuracy", "roc_auc", "precision_gecikme", "recall_gecikme", "f1_gecikme"]
    metric_labels = ["Accuracy", "ROC-AUC", "Precision(1)", "Recall(1)", "F1(1)"]
    xs = np.arange(len(metric_labels))
    width = 0.12
    for i, model_name in enumerate(cmp["model"]):
        vals = cmp.loc[cmp["model"] == model_name, metric_cols].iloc[0].tolist()
        axc.bar(xs + (i - 2.5) * width, vals, width, label=model_name)
    axc.set_xticks(xs, metric_labels)
    axc.set_ylim(0, 1.05)
    axc.set_title("Model Kiyaslama (hiperparametre sonrasi)")
    axc.legend(fontsize=7, loc="lower right", ncol=2)
    fig2.tight_layout()
    fig2.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
    if DESK_OUT.parent.exists():
        fig2.savefig(DESK_OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== KIYAS ===")
    print(cmp.to_string(index=False))
    print("\nKAYIT:", OUT / "hyperparam_comparison.png")


if __name__ == "__main__":
    main()
