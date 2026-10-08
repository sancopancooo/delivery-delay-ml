"""Final model: Bagging (tuned) + v2 ozellikler — kaydet."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_ready_v2.csv"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
DESK = Path.home() / "Desktop" / "delivery-delay-ml"
TARGET = "Late_delivery_risk"
CAT_COLS = [
    "Shipping Mode",
    "Category Name",
    "Order Region",
    "Customer Segment",
    "Department Name",
    "Market",
]


def mirror_file(src: Path):
    if not DESK.exists():
        return
    dest = DESK / src.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET].astype(int)
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=CAT_COLS, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=10,
            min_samples_split=5,
            random_state=42,
        ),
        n_estimators=100,
        max_samples=1.0,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    MODELS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    model_path = MODELS / "final_bagging_v2.joblib"
    columns_path = MODELS / "final_bagging_v2_columns.json"
    meta_path = MODELS / "final_bagging_v2_meta.json"

    joblib.dump(model, model_path)
    columns_path.write_text(
        json.dumps({"feature_columns": list(X.columns)}, indent=2), encoding="utf-8"
    )
    meta = {
        "model": "BaggingClassifier (DecisionTree)",
        "feature_set": "v2",
        "target": TARGET,
        "test_roc_auc": round(auc, 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "params": {
            "n_estimators": 100,
            "max_samples": 1.0,
            "max_features": 0.7,
            "estimator__max_depth": None,
            "estimator__min_samples_leaf": 10,
            "estimator__min_samples_split": 5,
            "estimator__class_weight": "balanced",
            "random_state": 42,
        },
        "raw_input_columns": [
            "Shipping Mode",
            "Days for shipment (scheduled)",
            "Category Name",
            "Order Region",
            "Customer Segment",
            "Department Name",
            "Market",
            "order_month",
            "order_weekday",
            "is_weekend",
            "scheduled_x_first",
            "scheduled_x_second",
        ],
        "files": {
            "model": str(model_path.name),
            "columns": str(columns_path.name),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = OUT / "FINAL_SUMMARY.md"
    summary.write_text(
        f"""# Final Model Ozeti

## Secilen model
- **BaggingClassifier** (Decision Tree estimator), hiperparametre tuned
- Ozellik seti: **v2**
- Test **ROC-AUC: {auc:.4f}**

## Neden bu model?
Bagging / Boosting / LR / RF / XGB kiyasinda v2 veride en yuksek AUC Bagging idi;
RandomizedSearch sonrasi {auc:.4f} seviyesine cikti.

## Leakage uyarisi
Modele dahil DEGIL: `Days for shipping (real)`, `Delivery Status`.

## Dosyalar
- `models/final_bagging_v2.joblib`
- `models/final_bagging_v2_columns.json`
- `models/final_bagging_v2_meta.json`
""",
        encoding="utf-8",
    )

    for p in [model_path, columns_path, meta_path, summary, DATA]:
        if p.exists():
            mirror_file(p)

    print("AUC", round(auc, 4))
    print("MODEL", model_path)
    print("META", meta_path)
    print("SUMMARY", summary)


if __name__ == "__main__":
    main()
