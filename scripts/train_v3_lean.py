"""v3 sade model: planlanan gun + etkilesim + kategori/bolge (+ kargo/segment)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_CANDIDATES = [
    ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv",
    Path.home() / "Desktop" / "delivery-delay-ml" / "data" / "raw" / "DataCoSupplyChainDataset.csv",
]
OUT = ROOT / "outputs"
DESK_OUT = Path.home() / "Desktop" / "delivery-delay-ml" / "outputs"
PROC = ROOT / "data" / "processed"
DESK_PROC = Path.home() / "Desktop" / "delivery-delay-ml" / "data" / "processed"
TARGET = "Late_delivery_risk"

# Budananlar: Department, Market, order_month/weekday/weekend
KEEP_CAT = ["Shipping Mode", "Category Name", "Order Region", "Customer Segment"]
KEEP_NUM = [
    "Days for shipment (scheduled)",
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


def mirror(path: Path):
    if DESK_OUT.parent.exists():
        DESK_OUT.mkdir(parents=True, exist_ok=True)
        (DESK_OUT / path.name).write_bytes(path.read_bytes())


def build_v3():
    raw_path = next(p for p in RAW_CANDIDATES if p.exists())
    df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    out = pd.DataFrame(
        {
            "Shipping Mode": df["Shipping Mode"],
            "Days for shipment (scheduled)": df["Days for shipment (scheduled)"],
            "Category Name": df["Category Name"],
            "Order Region": df["Order Region"],
            "Customer Segment": df["Customer Segment"],
            "scheduled_x_first": (df["Shipping Mode"] == "First Class").astype(int)
            * df["Days for shipment (scheduled)"],
            "scheduled_x_second": (df["Shipping Mode"] == "Second Class").astype(int)
            * df["Days for shipment (scheduled)"],
            TARGET: df[TARGET].astype(int),
        }
    )
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROC / "model_ready_v3.csv", index=False, encoding="utf-8-sig")
    if DESK_PROC.parent.exists():
        DESK_PROC.mkdir(parents=True, exist_ok=True)
        out.to_csv(DESK_PROC / "model_ready_v3.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    df = build_v3()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    X_ohe = pd.get_dummies(X, columns=KEEP_CAT, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_ohe, y, test_size=0.2, random_state=42, stratify=y
    )
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    # LR: scale numeric-ish full matrix lightly
    scaler = StandardScaler(with_mean=False)
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(Xtr_s, y_train)
    m_lr = metrics(y_test, lr.predict_proba(Xte_s)[:, 1])

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
    rf.fit(X_train, y_train)
    m_rf = metrics(y_test, rf.predict_proba(X_test)[:, 1])

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
    xgb.fit(X_train, y_train)
    m_xgb = metrics(y_test, xgb.predict_proba(X_test)[:, 1])

    cmp = pd.DataFrame(
        [
            {"version": "v1", "model": "XGBoost", "roc_auc": 0.7336, "note": "5 ham ozellik"},
            {"version": "v2", "model": "XGBoost", "roc_auc": 0.7482, "note": "tarih+dept+market+etkilesim"},
            {"version": "v3 lean", "model": "Logistic Regression", **m_lr, "note": "etkilesim+v1, zayiflar budandi"},
            {"version": "v3 lean", "model": "Random Forest", **m_rf, "note": "etkilesim+v1, zayiflar budandi"},
            {"version": "v3 lean", "model": "XGBoost", **m_xgb, "note": "etkilesim+v1, zayiflar budandi"},
        ]
    )
    OUT.mkdir(exist_ok=True)
    cmp.to_csv(OUT / "v3_lean_comparison.csv", index=False, encoding="utf-8-sig")
    mirror(OUT / "v3_lean_comparison.csv")

    payload = {
        "kept": KEEP_CAT + KEEP_NUM,
        "dropped_from_v2": [
            "Department Name",
            "Market",
            "order_month",
            "order_weekday",
            "is_weekend",
        ],
        "results": {"lr": m_lr, "rf": m_rf, "xgb": m_xgb},
        "v1_best_auc": 0.7336,
        "v2_best_auc": 0.7482,
        "v3_best_auc": max(m_lr["roc_auc"], m_rf["roc_auc"], m_xgb["roc_auc"]),
    }
    (OUT / "features_v3_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mirror(OUT / "features_v3_metrics.json")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = ["v1 XGB", "v2 XGB", "v3 LR", "v3 RF", "v3 XGB"]
    vals = [0.7336, 0.7482, m_lr["roc_auc"], m_rf["roc_auc"], m_xgb["roc_auc"]]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#54A24B", "#54A24B"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0.70, 0.78)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Sadelestirme deneyi: v1 vs v2 vs v3 lean")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "v3_lean_auc.png", dpi=150, bbox_inches="tight")
    mirror(OUT / "v3_lean_auc.png")
    plt.close()

    print("KOLONLAR:", list(df.columns))
    print("LR ", m_lr)
    print("RF ", m_rf)
    print("XGB", m_xgb)
    print("v2 best", 0.7482, "-> v3 best", payload["v3_best_auc"])
    print("KAYIT:", OUT / "v3_lean_auc.png")


if __name__ == "__main__":
    main()
