# Delivery Delay Prediction

Siparişin **gecikmeli teslim edilip edilmeyeceğini** tahmin eden ikili sınıflandırma projesi.

| | |
|--|--|
| **Hedef** | `Late_delivery_risk` (0/1) |
| **Veri** | [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (~180K) |
| **Final** | Bagging (tuned) + **v2** özellikler |
| **Test ROC-AUC** | **≈ 0.760** (leakage'siz) |

Tum deney listesi: [`EXPERIMENTS.md`](EXPERIMENTS.md)

---

## Neden 0.95+ notebook'lardan farkli?

Cok Kaggle notebook'u `Delivery Status` / gercek teslim gununu kullanir — bu **leakage**.  
Biz siparis aninda bilinen ozelliklerle kaldik; skor daha dusuk ama **gercekci**.

---

## Ozellikler (v2 final)

- Shipping Mode, planlanan gun, Category, Order Region, Customer Segment  
- Department, Market  
- Siparis ay / weekday / weekend  
- `scheduled_x_first`, `scheduled_x_second` (mod × gun)

---

## Ne denendi? (kisaca)

1. EDA + cleaning  
2. v1 → LR / RF / XGB / Decision Tree  
3. Hiperparametre (RandomizedSearch)  
4. v2 ozellik muhendisligi (+0.01~0.015 AUC)  
5. v3 budama (kotulesti → vazgecildi)  
6. Bagging vs Boosting → **Bagging kazandi**  
7. Bagging tune → **0.760**  
8. LightGBM / CatBoost / Voting  
9. Time-based split (~0.72)  
10. Geo denemesi (supheli sisme → finalde yok)  
11. Threshold optimizasyonu (F1)

Detayli tablo: `EXPERIMENTS.md` · grafikler: `outputs/`

---

## Kurulum

```powershell
cd $env:USERPROFILE\Projects\delivery-delay-ml
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Veri

```powershell
# %USERPROFILE%\.kaggle\kaggle.json gerekli
kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis -p data/raw --unzip
python scripts/build_features_v2.py
python scripts/save_final_model.py
```

### Demo UI (lokal deploy)

```powershell
streamlit run app.py
```

---

## Proje yapisi

```
app.py                 # Streamlit demo
EXPERIMENTS.md         # tum model/deney gunlugu
data/raw|processed/
scripts/               # egitim, tuning, kiyas
models/                # meta (+ joblib script ile)
outputs/               # metrikler, png
notebooks/
src/
```

---

## Teknik yigin

Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, matplotlib, Streamlit, Jupyter
