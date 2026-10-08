"""v2 ozellikler: tarih + Department + Market + mod x planlanan gun."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CANDIDATES = [
    ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv",
    Path.home() / "Desktop" / "delivery-delay-ml" / "data" / "raw" / "DataCoSupplyChainDataset.csv",
]
OUT = ROOT / "data" / "processed"
DESK = Path.home() / "Desktop" / "delivery-delay-ml" / "data" / "processed"
TARGET = "Late_delivery_risk"


def main():
    raw_path = next(p for p in RAW_CANDIDATES if p.exists())
    df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)

    order_date = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

    out = pd.DataFrame(
        {
            # v1
            "Shipping Mode": df["Shipping Mode"],
            "Days for shipment (scheduled)": df["Days for shipment (scheduled)"],
            "Category Name": df["Category Name"],
            "Order Region": df["Order Region"],
            "Customer Segment": df["Customer Segment"],
            # v2 guclu adaylar
            "Department Name": df["Department Name"],
            "Market": df["Market"],
            "order_month": order_date.dt.month,
            "order_weekday": order_date.dt.weekday,  # 0=Mon ... 6=Sun
            "is_weekend": (order_date.dt.weekday >= 5).astype(int),
            # etkileşim: hizli kargo + kisa sure = sikı vaat
            "scheduled_x_first": (
                (df["Shipping Mode"] == "First Class").astype(int)
                * df["Days for shipment (scheduled)"]
            ),
            "scheduled_x_second": (
                (df["Shipping Mode"] == "Second Class").astype(int)
                * df["Days for shipment (scheduled)"]
            ),
            TARGET: df[TARGET].astype(int),
        }
    )

    before = len(out)
    out = out.dropna().reset_index(drop=True)
    print("ham:", before, "temiz:", len(out), "dusen:", before - len(out))
    print("eksik kalan:", int(out.isna().sum().sum()))
    print("kolonlar:", list(out.columns))
    print(out.head(3).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "model_ready_v2.csv", index=False, encoding="utf-8-sig")
    if DESK.parent.exists():
        DESK.mkdir(parents=True, exist_ok=True)
        out.to_csv(DESK / "model_ready_v2.csv", index=False, encoding="utf-8-sig")
    print("KAYIT:", OUT / "model_ready_v2.csv")


if __name__ == "__main__":
    main()
