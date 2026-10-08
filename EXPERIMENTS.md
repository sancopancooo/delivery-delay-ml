# Deney Gunlugu — Delivery Delay Prediction

Bu dokuman projede denenen **tum modelleri, ozellik setlerini ve dersleri** kaydeder.
Amac: skor avlamak degil; leakage'siz, anlatilabilir bir pipeline.

**Veri:** DataCo Smart Supply Chain (~180.519 satir)  
**Hedef:** `Late_delivery_risk` (0/1)

---

## 0) Leakage kurallari (bastan sabit)

Modele **asla** girmedi:
- `Days for shipping (real)`
- `Delivery Status` (hedefle neredeyse birebir)

Odeme tipi (`Type`) bilerek cikarildi (nedensel hikaye lojistik).

---

## 1) Ozellik setleri

### v1
`Shipping Mode`, `Days for shipment (scheduled)`, `Category Name`, `Order Region`, `Customer Segment`

### v2 (final ozellik seti)
v1 + `Department Name`, `Market`, `order_month`, `order_weekday`, `is_weekend`,  
`scheduled_x_first`, `scheduled_x_second`

### v3 lean
v1 + sadece etkilesimler (tarih/dept/market budandi) → skor dustu, vazgecildi

### v4 geo (denendi, final degil)
`cross_country`, `geo_dist_km`  
**Uyari:** Random splitte AUC sismesi (~0.88) goruldu; konum surekli degiskenler musteri kimligi gibi ezberlenebilir. Final modele **alinmadi**.

---

## 2) Model kiyaslari (v1 / v2, random stratified %80/%20)

| Model | Ozellik | ROC-AUC | Not |
|--------|---------|---------|-----|
| Logistic Regression | v1 | 0.724 | Baseline |
| Decision Tree (default) | v1 | 0.724 | |
| Decision Tree (tuned) | v1 | 0.734 | RandomizedSearch |
| Random Forest | v1 | 0.733 | |
| Random Forest (tuned) | v1 | 0.733 | |
| XGBoost | v1 | 0.731 | |
| XGBoost (tuned) | v1 | 0.734 | |
| Logistic Regression | v2 | 0.729 | |
| Random Forest | v2 | 0.744 | |
| XGBoost | v2 | 0.748 | |
| Bagging (agac) | v2 | 0.756 | Bagging>Boosting bu veride |
| **Bagging (tuned)** | **v2** | **0.760** | **Final model** |
| LightGBM | v2 | 0.752 | Ek deney |
| CatBoost | v2 | 0.755 | Ek deney |
| Voting (Bag+RF+XGB) | v2 | 0.756 | Ensemble; finali gecemedi |

---

## 3) Bagging vs Boosting (v2)

Sira: Bagging → RF → Extra Trees → AdaBoost → GradientBoosting → XGBoost

En iyi aile uyesi once Bagging (~0.756), tuning sonrasi **0.760**.

---

## 4) Hiperparametre

Yontem: `RandomizedSearchCV`, scoring=`roc_auc`, CV=3

- Decision Tree: `max_depth=6`, `min_samples_leaf=10`, ...
- RF / XGB: kucuk kazanc (+0.001~0.003)
- Bagging final: `n_estimators=100`, `max_features=0.7`, `min_samples_leaf=10`, `max_depth=None`

**Ders:** Bu projede tuning < ozellik muhendisligi.

---

## 5) Feature importance (XGB v2)

1. Planlanan gun  
2. `scheduled_x_first` / `scheduled_x_second` (v2'nin asil kazanci)  
3. Kategori / bolge (parca parca)  
Tarih / department / market tek tek zayif ama budanınca (v3) toplam skor dustu.

---

## 6) Ek validasyonlar

| Deney | ROC-AUC | Yorum |
|--------|---------|--------|
| Time-based split (Bagging v2) | ~0.72 | Daha zor / daha gercekci |
| Threshold opt (F1 max) | esik≈0.32 | AUC ayni; recall↑ precision↓ |
| Geo v4 | ~0.88 (supheli) | Finalde yok; ezber riski |

---

## 7) Final karar

- **Model:** `BaggingClassifier` (tuned)  
- **Ozellik:** v2  
- **Test ROC-AUC:** **0.760**  
- Dosya: `models/final_bagging_v2.joblib` (script ile uretilir)  
- Meta: `models/final_bagging_v2_meta.json`

---

## 8) "Biri 0.85+ bulursa?"

Leakage'siz ayni protokolle 0.85 **sasirtici** olur.  
`Delivery Status` / gercek teslim gunu girmisse sasirmayiz — skor yapay sisir.

---

## Script indeksi

| Script | Ne yapar |
|--------|----------|
| `scripts/build_features_v2.py` | v2 tablo |
| `scripts/train_v2_compare.py` | v1 vs v2 |
| `scripts/train_v3_lean.py` | budama deneyi |
| `scripts/train_bagging_boosting.py` | aile kiyasi |
| `scripts/tune_bagging.py` / `tune_rf_xgb.py` | hiperparametre |
| `scripts/feature_importance_v2.py` | importance |
| `scripts/run_remaining_experiments.py` | LGBM/CatBoost/voting/geo/time/threshold |
| `scripts/save_final_model.py` | final kayit |
