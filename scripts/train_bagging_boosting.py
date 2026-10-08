"""Bagging vs Boosting ailesi — v2 ozelliklerle."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    models = {
        # --- BAGGING ailesi ---
        "Bagging (agac)": (
            "bagging",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=12, class_weight="balanced", random_state=42
                ),
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "Random Forest": (
            "bagging",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "Extra Trees": (
            "bagging",
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        # --- BOOSTING ailesi ---
        "AdaBoost": (
            "boosting",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced"),
                n_estimators=100,
                learning_rate=0.5,
                random_state=42,
            ),
        ),
        "Gradient Boosting": (
            "boosting",
            GradientBoostingClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.08,
                random_state=42,
            ),
        ),
        "XGBoost": (
            "boosting",
            XGBClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                min_child_weight=3,
                reg_lambda=1.0,
                scale_pos_weight=neg / pos,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }

    rows = []
    for name, (family, model) in models.items():
        print(f">>> {family.upper()} | {name}")
        t0 = perf_counter()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        m = metrics(y_test, proba)
        m.update(
            {
                "model": name,
                "family": family,
                "seconds": round(perf_counter() - t0, 2),
            }
        )
        rows.append(m)
        print("   AUC", m["roc_auc"], "acc", m["accuracy"], f"({m['seconds']}s)")

    cmp = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    OUT.mkdir(exist_ok=True)
    cmp.to_csv(OUT / "bagging_vs_boosting.csv", index=False, encoding="utf-8-sig")
    mirror(OUT / "bagging_vs_boosting.csv")

    payload = {
        "idea": {
            "bagging": "Ayni tur zayif modelleri paralel egit, oyla (varyansi dusurur).",
            "boosting": "Hatalari sirayla duzeltmeye calis (bias'i dusurur); genelde guclu.",
        },
        "best": cmp.iloc[0].to_dict(),
        "results": cmp.to_dict(orient="records"),
    }
    (OUT / "bagging_vs_boosting.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mirror(OUT / "bagging_vs_boosting.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#4C78A8" if f == "bagging" else "#F58518" for f in cmp["family"]]
    axes[0].barh(cmp["model"], cmp["roc_auc"], color=colors)
    axes[0].set_xlim(0.70, 0.78)
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_title("Bagging (mavi) vs Boosting (turuncu)")
    for i, v in enumerate(cmp["roc_auc"]):
        axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)

    fam = cmp.groupby("family")["roc_auc"].max().reindex(["bagging", "boosting"])
    axes[1].bar(fam.index, fam.values, color=["#4C78A8", "#F58518"])
    axes[1].set_ylim(0.70, 0.78)
    axes[1].set_ylabel("Aile ici en iyi ROC-AUC")
    axes[1].set_title("Aile ozeti")
    for i, v in enumerate(fam.values):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center")

    fig.suptitle("Bagging vs Boosting — v2 veri", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "bagging_vs_boosting.png", dpi=150, bbox_inches="tight")
    mirror(OUT / "bagging_vs_boosting.png")
    plt.close()

    print("\n=== SIRALAMA ===")
    print(cmp[["family", "model", "roc_auc", "accuracy", "f1_gecikme", "seconds"]].to_string(index=False))
    print("KAYIT:", OUT / "bagging_vs_boosting.png")


if __name__ == "__main__":
    main()
