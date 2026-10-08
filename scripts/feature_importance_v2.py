"""XGBoost v2 feature importance."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

# ham ozellik aileleri (one-hot sonrasi eslestirme)
FAMILY = {
    "Shipping Mode": "v1_ShippingMode",
    "Days for shipment (scheduled)": "v1_ScheduledDays",
    "Category Name": "v1_Category",
    "Order Region": "v1_OrderRegion",
    "Customer Segment": "v1_Segment",
    "Department Name": "v2_Department",
    "Market": "v2_Market",
    "order_month": "v2_Date",
    "order_weekday": "v2_Date",
    "is_weekend": "v2_Date",
    "scheduled_x_first": "v2_Interaction",
    "scheduled_x_second": "v2_Interaction",
}


def family_of(col: str) -> str:
    if col in FAMILY:
        return FAMILY[col]
    for prefix, fam in FAMILY.items():
        if col.startswith(prefix + "_"):
            return fam
    return "other"


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

    model = XGBClassifier(
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
    model.fit(X_train, y_train)

    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp["family"] = imp["feature"].map(family_of)
    imp["is_v2"] = imp["family"].str.startswith("v2_")

    OUT.mkdir(exist_ok=True)
    imp.to_csv(OUT / "feature_importance_v2.csv", index=False, encoding="utf-8-sig")
    mirror(OUT / "feature_importance_v2.csv")

    # top 15
    top = imp.head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    colors = ["#F58518" if v else "#4C78A8" for v in top["is_v2"]]
    axes[0].barh(top["feature"], top["importance"], color=colors)
    axes[0].set_title("En onemli 15 ozellik (turuncu = v2)")
    axes[0].set_xlabel("XGBoost importance (gain)")

    fam = (
        imp.groupby("family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=True)
    )
    fam_colors = ["#F58518" if f.startswith("v2_") else "#4C78A8" for f in fam["family"]]
    axes[1].barh(fam["family"], fam["importance"], color=fam_colors)
    axes[1].set_title("Ozellik ailesi toplami")
    axes[1].set_xlabel("Toplam importance")

    fig.suptitle("Feature Importance — XGBoost v2", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance_v2.png", dpi=150, bbox_inches="tight")
    mirror(OUT / "feature_importance_v2.png")
    plt.close()

    print("=== TOP 15 ===")
    print(imp.head(15)[["feature", "importance", "family"]].to_string(index=False))
    print("\n=== AILE TOPLAMI ===")
    print(fam.sort_values("importance", ascending=False).to_string(index=False))
    print("\nv2 payi:", round(imp.loc[imp["is_v2"], "importance"].sum(), 4))
    print("v1 payi:", round(imp.loc[~imp["is_v2"], "importance"].sum(), 4))
    print("KAYIT:", OUT / "feature_importance_v2.png")


if __name__ == "__main__":
    main()
