"""BaggingClassifier hiperparametre aramasi — v2."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

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


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET].astype(int)
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=CAT_COLS, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # onceki bagging (referans)
    prev = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    prev.fit(X_train, y_train)
    m_prev = metrics(y_test, prev.predict_proba(X_test)[:, 1])

    print(">>> Bagging RandomizedSearch...")
    t0 = perf_counter()
    base = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced", random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    space = {
        "n_estimators": [50, 100, 150, 200],
        "max_samples": [0.6, 0.8, 1.0],
        "max_features": [0.5, 0.7, 1.0],
        "estimator__max_depth": [8, 10, 12, 16, 20, None],
        "estimator__min_samples_leaf": [1, 5, 10, 20],
        "estimator__min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        base,
        param_distributions=space,
        n_iter=18,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    m_best = metrics(y_test, best.predict_proba(X_test)[:, 1])
    elapsed = round(perf_counter() - t0, 1)

    payload = {
        "previous": {"params_note": "n_estimators=100, max_depth=12", "test": m_prev},
        "tuned": {
            "best_params": search.best_params_,
            "best_cv_auc": round(float(search.best_score_), 4),
            "test": m_best,
            "seconds": elapsed,
        },
        "leaderboard_context": {
            "prev_bagging_auc": m_prev["roc_auc"],
            "xgb_v2_auc": 0.7482,
        },
    }
    OUT.mkdir(exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    (OUT / "bagging_tuned.json").write_text(text, encoding="utf-8")
    if DESK_OUT.parent.exists():
        DESK_OUT.mkdir(parents=True, exist_ok=True)
        (DESK_OUT / "bagging_tuned.json").write_text(text, encoding="utf-8")

    cmp = pd.DataFrame(
        [
            {"model": "Bagging (onceki)", **m_prev},
            {"model": "Bagging (tuned)", **m_best},
            {
                "model": "XGBoost v2",
                "accuracy": 0.6994,
                "roc_auc": 0.7482,
                "precision_gecikme": 0.8486,
                "recall_gecikme": 0.5498,
                "f1_gecikme": 0.6673,
            },
        ]
    )
    cmp.to_csv(OUT / "bagging_tuned_comparison.csv", index=False, encoding="utf-8-sig")
    mirror(OUT / "bagging_tuned_comparison.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = cmp["model"]
    vals = cmp["roc_auc"]
    colors = ["#4C78A8", "#54A24B", "#F58518"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0.72, 0.78)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Bagging hiperparametre: onceki vs tuned")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "bagging_tuned.png", dpi=150, bbox_inches="tight")
    mirror(OUT / "bagging_tuned.png")
    plt.close()

    print("Onceki:", m_prev)
    print("Best params:", search.best_params_)
    print("CV AUC:", round(float(search.best_score_), 4))
    print("Tuned test:", m_best)
    print("Sure:", elapsed, "s")
    print("KAYIT:", OUT / "bagging_tuned.png")


if __name__ == "__main__":
    main()
