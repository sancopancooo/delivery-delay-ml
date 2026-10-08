# Final Model Ozeti

## Secilen model
- **BaggingClassifier** (Decision Tree estimator), hiperparametre tuned
- Ozellik seti: **v2**
- Test **ROC-AUC: 0.7598**

## Neden bu model?
Bagging / Boosting / LR / RF / XGB kiyasinda v2 veride en yuksek AUC Bagging idi;
RandomizedSearch sonrasi 0.7598 seviyesine cikti.

## Leakage uyarisi
Modele dahil DEGIL: `Days for shipping (real)`, `Delivery Status`.

## Dosyalar
- `models/final_bagging_v2.joblib`
- `models/final_bagging_v2_columns.json`
- `models/final_bagging_v2_meta.json`
