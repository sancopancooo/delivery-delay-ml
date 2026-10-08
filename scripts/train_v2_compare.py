"""v2 ozelliklerle LR / RF / XGB — v1 ile kiyas."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_ready_v2.csv"
OUT = ROOT / "outputs"
DESK_OUT = Path.home() / "Desktop" / "delivery-delay-ml" / "outputs"
TARGET = "Late_delivery_risk"

CAT_COLS = [
    "Shipping Mode",
    "Category Name",
    "Order Region",
    "Customer Segment",
    "Department Name",
    "Market",
]
NUM_COLS = [
    "Days for shipment (scheduled)",
    "order_month",
    "order_weekday",
    "is_weekend",
    "scheduled_x_first",
    "scheduled_x_second",
]


def metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "precision_gecikme": round(float(precision_score(y_true, pred)), 4),
        "recall_gecikme": round(float(recall_score(y_true, pred)), 4),
        "f1_gecikme": round(float(f1_score(y_true, pred)), 4),
    }


def mirror_save(path: Path):
    if DESK_OUT.parent.exists():
        DESK_OUT.mkdir(parents=True, exist_ok=True)
        dest = DESK_OUT / path.name
        dest.write_bytes(path.read_bytes())


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    # One-hot + scale for LR
    pre_lr = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )
    lr = Pipeline(
        [
            ("prep", pre_lr),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    # Trees: pandas get_dummies
    X_ohe = pd.get_dummies(X, columns=CAT_COLS, dtype=int)
    Xtr_ohe, Xte_ohe, ytr, yte = train_test_split(
        X_ohe, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=5,
        min_samples_split=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    print("LR fit...")
    lr.fit(X_train, y_train)
    m_lr = metrics(y_test, lr.predict_proba(X_test)[:, 1])

    print("RF fit...")
    rf.fit(Xtr_ohe, ytr)
    m_rf = metrics(yte, rf.predict_proba(Xte_ohe)[:, 1])

    print("XGB fit...")
    xgb.fit(Xtr_ohe, ytr)
    m_xgb = metrics(yte, xgb.predict_proba(Xte_ohe)[:, 1])

    rows = [
        {"version": "v1", "model": "Logistic Regression", "roc_auc": 0.7239, "accuracy": 0.6940, "f1_gecikme": 0.6600},
        {"version": "v1", "model": "Random Forest (tuned)", "roc_auc": 0.7329, "accuracy": 0.6968, "f1_gecikme": 0.6627},
        {"version": "v1", "model": "XGBoost (tuned)", "roc_auc": 0.7336, "accuracy": 0.6970, "f1_gecikme": 0.6635},
        {"version": "v2", "model": "Logistic Regression", **m_lr},
        {"version": "v2", "model": "Random Forest (tuned params)", **m_rf},
        {"version": "v2", "model": "XGBoost (tuned params)", **m_xgb},
    ]
    # normalize columns
    cmp = pd.DataFrame(rows)
    for c in ["accuracy", "roc_auc", "precision_gecikme", "recall_gecikme", "f1_gecikme"]:
        if c not in cmp.columns:
            cmp[c] = np.nan

    OUT.mkdir(exist_ok=True)
    cmp.to_csv(OUT / "v1_vs_v2_comparison.csv", index=False, encoding="utf-8-sig")
    mirror_save(OUT / "v1_vs_v2_comparison.csv")

    payload = {
        "new_features": [
            "Department Name",
            "Market",
            "order_month",
            "order_weekday",
            "is_weekend",
            "scheduled_x_first",
            "scheduled_x_second",
        ],
        "v2_results": {
            "logistic_regression": m_lr,
            "random_forest": m_rf,
            "xgboost": m_xgb,
        },
        "v1_best_auc": 0.7336,
        "v2_best_auc": max(m_lr["roc_auc"], m_rf["roc_auc"], m_xgb["roc_auc"]),
    }
    (OUT / "features_v2_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mirror_save(OUT / "features_v2_metrics.json")

    # chart
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["LR", "RF", "XGB"]
    v1 = [0.7239, 0.7329, 0.7336]
    v2 = [m_lr["roc_auc"], m_rf["roc_auc"], m_xgb["roc_auc"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, v1, 0.4, label="v1 ozellikler")
    ax.bar(x + 0.2, v2, 0.4, label="v2 + guclu adaylar")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.70, 0.80)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Ozellik muhendisligi: v1 vs v2")
    ax.legend()
    for i, (a, b) in enumerate(zip(v1, v2)):
        ax.text(i - 0.2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.002, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b - 0.012, f"{b-a:+.3f}", ha="center", fontsize=8, color="darkgreen" if b >= a else "darkred")
    fig.tight_layout()
    fig.savefig(OUT / "v1_vs_v2_auc.png", dpi=150, bbox_inches="tight")
    mirror_save(OUT / "v1_vs_v2_auc.png")
    plt.close()

    print("\n=== v2 SONUCLAR ===")
    print("LR ", m_lr)
    print("RF ", m_rf)
    print("XGB", m_xgb)
    print("v1 best AUC:", 0.7336, "-> v2 best:", payload["v2_best_auc"])
    print("KAYIT:", OUT / "v1_vs_v2_auc.png")


if __name__ == "__main__":
    main()
