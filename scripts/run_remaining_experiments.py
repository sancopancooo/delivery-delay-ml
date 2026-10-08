"""Eksik alanlar (temiz): geo v4, LightGBM, CatBoost, voting, threshold, time-split."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_CANDIDATES = [
    ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv",
    Path.home() / "Desktop" / "delivery-delay-ml" / "data" / "raw" / "DataCoSupplyChainDataset.csv",
]
V2 = ROOT / "data" / "processed" / "model_ready_v2.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TARGET = "Late_delivery_risk"
CAT = [
    "Shipping Mode",
    "Category Name",
    "Order Region",
    "Customer Segment",
    "Department Name",
    "Market",
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def make_bagging():
    return BaggingClassifier(
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


def auc(y, proba):
    return round(float(roc_auc_score(y, proba)), 4)


def best_threshold(y, proba):
    best = None
    rows = []
    for t in np.linspace(0.2, 0.8, 31):
        pred = (proba >= t).astype(int)
        row = {
            "threshold": round(float(t), 3),
            "f1": round(float(f1_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        }
        rows.append(row)
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best, pd.DataFrame(rows)


def main():
    v2 = pd.read_csv(V2)
    y = v2[TARGET].astype(int)
    X = pd.get_dummies(v2.drop(columns=[TARGET]), columns=CAT, dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # A) Bagging v2 (final referans)
    bag = make_bagging()
    bag.fit(Xtr, ytr)
    proba = bag.predict_proba(Xte)[:, 1]
    results["bagging_v2"] = {"roc_auc": auc(yte, proba)}
    thr, thr_df = best_threshold(yte.to_numpy(), proba)
    thr_df.to_csv(OUT / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    results["threshold_opt"] = {
        **thr,
        "note": "Esik 0.5 yerine F1 maximize; AUC degismez, precision/recall dengesi degisir",
    }

    # B) LightGBM / CatBoost
    try:
        from lightgbm import LGBMClassifier

        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb.fit(Xtr, ytr)
        results["lightgbm_v2"] = {"roc_auc": auc(yte, lgb.predict_proba(Xte)[:, 1]), "status": "ok"}
    except Exception as e:
        results["lightgbm_v2"] = {"status": "skip", "error": str(e)[:200]}

    try:
        from catboost import CatBoostClassifier

        cat = CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
        )
        # ayni one-hot matris
        cat.fit(Xtr, ytr)
        results["catboost_v2"] = {"roc_auc": auc(yte, cat.predict_proba(Xte)[:, 1]), "status": "ok"}
    except Exception as e:
        results["catboost_v2"] = {"status": "skip", "error": str(e)[:200]}

    # C) Voting
    neg, pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    vote = VotingClassifier(
        estimators=[
            ("bag", make_bagging()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=16,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.05,
                    min_child_weight=3,
                    scale_pos_weight=neg / max(pos, 1),
                    eval_metric="auc",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        voting="soft",
        n_jobs=-1,
    )
    vote.fit(Xtr, ytr)
    results["voting_bag_rf_xgb"] = {
        "roc_auc": auc(yte, vote.predict_proba(Xte)[:, 1]),
        "members": ["Bagging", "RF", "XGB"],
    }

    # D) Time-based split + E) geo — raw gerekir
    raw_path = next(p for p in RAW_CANDIDATES if p.exists())
    raw = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    order_date = pd.to_datetime(raw["order date (DateOrders)"], errors="coerce")
    base = pd.DataFrame(
        {
            "Shipping Mode": raw["Shipping Mode"],
            "Days for shipment (scheduled)": raw["Days for shipment (scheduled)"],
            "Category Name": raw["Category Name"],
            "Order Region": raw["Order Region"],
            "Customer Segment": raw["Customer Segment"],
            "Department Name": raw["Department Name"],
            "Market": raw["Market"],
            "order_month": order_date.dt.month,
            "order_weekday": order_date.dt.weekday,
            "is_weekend": (order_date.dt.weekday >= 5).astype(int),
            "scheduled_x_first": (raw["Shipping Mode"] == "First Class").astype(int)
            * raw["Days for shipment (scheduled)"],
            "scheduled_x_second": (raw["Shipping Mode"] == "Second Class").astype(int)
            * raw["Days for shipment (scheduled)"],
            TARGET: raw[TARGET].astype(int),
            "order_date": order_date,
            "Latitude": raw["Latitude"],
            "Longitude": raw["Longitude"],
            "Order Country": raw["Order Country"],
            "Customer Country": raw["Customer Country"],
        }
    ).dropna(subset=[TARGET, "order_date"]).reset_index(drop=True)

    # time split
    base_sorted = base.sort_values("order_date")
    cut = int(len(base_sorted) * 0.8)
    tr_f, te_f = base_sorted.iloc[:cut], base_sorted.iloc[cut:]
    feat_cols = [c for c in v2.columns if c != TARGET]
    Xtr_t = pd.get_dummies(tr_f[feat_cols], columns=CAT, dtype=int)
    Xte_t = pd.get_dummies(te_f[feat_cols], columns=CAT, dtype=int)
    Xte_t = Xte_t.reindex(columns=Xtr_t.columns, fill_value=0)
    bag_t = make_bagging()
    bag_t.fit(Xtr_t, tr_f[TARGET].astype(int))
    results["bagging_time_split"] = {
        "roc_auc": auc(te_f[TARGET].astype(int), bag_t.predict_proba(Xte_t)[:, 1]),
        "train_max_date": str(tr_f["order_date"].max()),
        "test_min_date": str(te_f["order_date"].min()),
        "note": "Zaman bazli split; random splitten genelde daha zor / daha gercekci",
    }

    # geo v4 — country centroid ONLY from train fold (no test leakage)
    tr_part, te_part = train_test_split(
        base, test_size=0.2, random_state=42, stratify=base[TARGET]
    )
    tr_part = tr_part.copy()
    te_part = te_part.copy()
    cent = tr_part.groupby("Order Country")[["Latitude", "Longitude"]].mean()
    cent.columns = ["oc_lat", "oc_lon"]

    def add_geo(part):
        p = part.merge(cent, left_on="Order Country", right_index=True, how="left")
        p["cross_country"] = (
            p["Order Country"].astype(str) != p["Customer Country"].astype(str)
        ).astype(int)
        p["geo_dist_km"] = haversine_km(
            p["Latitude"].to_numpy(),
            p["Longitude"].to_numpy(),
            p["oc_lat"].fillna(p["Latitude"]).to_numpy(),
            p["oc_lon"].fillna(p["Longitude"]).to_numpy(),
        )
        med = float(np.nanmedian(p["geo_dist_km"]))
        p["geo_dist_km"] = p["geo_dist_km"].replace([np.inf, -np.inf], np.nan).fillna(med)
        return p

    tr_g = add_geo(tr_part)
    te_g = add_geo(te_part)
    g_cols = feat_cols + ["cross_country", "geo_dist_km"]
    Xtr_geo = pd.get_dummies(tr_g[g_cols], columns=CAT, dtype=int)
    Xte_geo = pd.get_dummies(te_g[g_cols], columns=CAT, dtype=int)
    Xte_geo = Xte_geo.reindex(columns=Xtr_geo.columns, fill_value=0)
    bag_g = make_bagging()
    bag_g.fit(Xtr_geo, tr_g[TARGET].astype(int))
    results["bagging_v4_geo"] = {
        "roc_auc": auc(te_g[TARGET].astype(int), bag_g.predict_proba(Xte_geo)[:, 1]),
        "added": ["cross_country", "geo_dist_km"],
        "note": "Ulke centroid sadece train'den; lat/lon ham olarak modele girmedi",
    }

    # chart
    plot_items = [
        ("Bagging v2", results["bagging_v2"]["roc_auc"]),
        ("LightGBM", results.get("lightgbm_v2", {}).get("roc_auc")),
        ("CatBoost", results.get("catboost_v2", {}).get("roc_auc")),
        ("Voting", results["voting_bag_rf_xgb"]["roc_auc"]),
        ("Geo v4", results["bagging_v4_geo"]["roc_auc"]),
        ("Time-split", results["bagging_time_split"]["roc_auc"]),
    ]
    labels = [a for a, b in plot_items if b is not None]
    vals = [b for a, b in plot_items if b is not None]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, vals, color="#4C78A8")
    ax.set_ylim(0.65, max(vals) + 0.04)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Eksik deneyler (temiz skorlar)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "remaining_experiments.png", dpi=150, bbox_inches="tight")
    plt.close()

    (OUT / "remaining_experiments.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
