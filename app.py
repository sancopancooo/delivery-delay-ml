"""Portföy demo — lokal: streamlit run app.py | Cloud: share.streamlit.io"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
META = ROOT / "models" / "final_bagging_v2_meta.json"
EXP = ROOT / "EXPERIMENTS.md"

st.set_page_config(page_title="Delivery Delay ML", page_icon="📦", layout="wide")

st.title("Teslimat Gecikmesi Tahmini")
st.markdown(
    "**Portföy demo** · DataCo Smart Supply Chain · ikili sınıflandırma (`Late_delivery_risk`)"
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Final model", "Bagging (tuned)")
with c2:
    st.metric("Test ROC-AUC", "0.760")
with c3:
    st.metric("Özellik seti", "v2 (leakage'siz)")

st.info(
    "Leakage yok: `Days for shipping (real)` ve `Delivery Status` modele girmedi. "
    "Kaggle'da %90+ gören notebook'lar genelde bu kolonları kullanır."
)

tab1, tab2, tab3, tab4 = st.tabs(["Özet", "Grafikler", "Kıyas tablosu", "Deney günlüğü"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Final model meta")
        if META.exists():
            st.json(json.loads(META.read_text(encoding="utf-8")))
        else:
            st.warning("Meta dosyası yok.")
    with right:
        st.subheader("Ne öğrendik?")
        st.markdown(
            """
- Baseline → RF/XGB → Bagging/Boosting kıyası  
- Hiperparametre küçük kazanç; **özellik mühendisliği** daha çok işe yaradı  
- Bu veride **Bagging**, XGBoost'u geçti  
- Geo özelliği random splitte şişebiliyor → finalde yok  
- Time-split ~0.72 (daha gerçekçi / daha zor)
            """
        )
        st.link_button(
            "GitHub repo",
            "https://github.com/sancopancooo/delivery-delay-ml",
        )

with tab2:
    charts = [
        ("Bagging tune", OUT / "bagging_tuned.png"),
        ("Bagging vs Boosting", OUT / "bagging_vs_boosting.png"),
        ("v1 vs v2 özellikler", OUT / "v1_vs_v2_auc.png"),
        ("Feature importance", OUT / "feature_importance_v2.png"),
        ("Ek deneyler", OUT / "remaining_experiments.png"),
        ("Model kıyas", OUT / "model_comparison.png"),
    ]
    cols = st.columns(2)
    for i, (title, path) in enumerate(charts):
        with cols[i % 2]:
            st.caption(title)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.write(f"Yok: `{path.name}`")

with tab3:
    cmp = OUT / "model_comparison.csv"
    if cmp.exists():
        st.dataframe(pd.read_csv(cmp), use_container_width=True)
    else:
        st.write("Kıyas CSV yok.")
    rem = OUT / "remaining_experiments.json"
    if rem.exists():
        st.subheader("Ek deney JSON")
        st.json(json.loads(rem.read_text(encoding="utf-8")))

with tab4:
    if EXP.exists():
        st.markdown(EXP.read_text(encoding="utf-8"))
    else:
        st.write("EXPERIMENTS.md yok.")

st.markdown("---")
st.caption("Kaynak: github.com/sancopancooo/delivery-delay-ml")
