"""Basit demo UI — lokal calistir: streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
META = ROOT / "models" / "final_bagging_v2_meta.json"
SUMMARY = ROOT / "outputs" / "FINAL_SUMMARY.md"
CMP = ROOT / "outputs" / "model_comparison.csv"
EXP = ROOT / "EXPERIMENTS.md"

st.set_page_config(page_title="Delivery Delay ML", layout="wide")
st.title("Teslimat Gecikmesi — Portföy Demo")
st.caption("Final: Bagging (tuned) + v2 özellikler · ROC-AUC ≈ 0.76 (leakage'siz)")

col1, col2 = st.columns(2)
with col1:
    if META.exists():
        import json

        meta = json.loads(META.read_text(encoding="utf-8"))
        st.subheader("Final model")
        st.json(meta)
    else:
        st.info("Meta yok — `python scripts/save_final_model.py` çalıştır.")

with col2:
    if SUMMARY.exists():
        st.subheader("Özet")
        st.markdown(SUMMARY.read_text(encoding="utf-8"))

st.subheader("Model kıyas dosyası")
if CMP.exists():
    st.dataframe(pd.read_csv(CMP), use_container_width=True)
else:
    st.write("`outputs/model_comparison.csv` henüz yok.")

if EXP.exists():
    with st.expander("Tüm deney günlüğü (EXPERIMENTS.md)"):
        st.markdown(EXP.read_text(encoding="utf-8"))

st.markdown("---")
st.markdown(
    "Leakage yok: `Days for shipping (real)` ve `Delivery Status` modele girmedi. "
    "Detay: `README.md` + `EXPERIMENTS.md`."
)
