import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Prediksi", page_icon="🤖", layout="wide")

with st.sidebar:
    st.markdown("### 🎓 Prediksi Kelulusan")
    st.markdown("---")
    st.page_link("app.py",                    label="🏠 Beranda")
    st.page_link("pages/1_EDA.py",            label="📊 EDA & Analisis Data")
    st.page_link("pages/2_Prediksi.py",       label="🤖 Prediksi Mahasiswa")
    st.page_link("pages/3_Evaluasi_Model.py", label="📈 Evaluasi Model")
    st.page_link("pages/4_Interpretasi.py",   label="🔍 Interpretasi SHAP")
    st.page_link("pages/5_Dokumentasi.py",    label="📋 Dokumentasi")

@st.cache_resource
def load_model():
    lr  = joblib.load("../models/logistic_regression.pkl")
    rf  = joblib.load("../models/best_model.pkl")
    info = joblib.load("../models/feature_info.pkl")
    return lr, rf, info

model_lr, model_rf, feat_info = load_model()
FEATURES = feat_info["num"] + feat_info["cat"]

st.title("🤖 Prediksi Kelulusan Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi kemungkinan lulus tepat waktu.")

col_form, col_result = st.columns([2, 1])

with col_form:
    st.markdown("### 📝 Data Mahasiswa")

    with st.expander("📚 Data Akademik", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ipk = st.slider("IPK", 1.5, 4.0, 3.0, 0.01)
            sks = st.slider("SKS Lulus", 20, 160, 110, 1)
            kehadiran = st.slider("Kehadiran (%)", 40.0, 100.0, 85.0, 0.5)
        with c2:
            nilai_rata = st.slider("Nilai Rata-rata", 50.0, 100.0, 78.0, 0.5)
            jumlah_cuti = st.number_input("Jumlah Semester Cuti", 0, 6, 0)
            jumlah_mengulang = st.number_input("Jumlah MK Diulang", 0, 10, 1)

    with st.expander("👤 Data Personal & Sosial", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            lama_studi = st.slider("Lama Studi (semester)", 6, 14, 8)
            penghasilan = st.selectbox("Penghasilan Orang Tua", ["Rendah","Menengah","Tinggi"])
        with c4:
            st.markdown("**Status Kerja**")
            kerja_label = st.radio("", ["Tidak Bekerja","Part-time","Full-time"], horizontal=True)
            status_kerja = {"Tidak Bekerja":0,"Part-time":1,"Full-time":2}[kerja_label]

            jalur = st.selectbox("Jalur Masuk", ["SNBP","SNBT","Mandiri"])
            org = 1 if st.checkbox("Aktif Organisasi") else 0

    model_choice = st.selectbox("🤖 Pilih Model", ["Logistic Regression","Random Forest"])
    predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

with col_result:
    st.markdown("### 🎯 Hasil Prediksi")

    if predict_btn:
        input_data = pd.DataFrame([{
            "ipk": ipk, "sks": sks, "kehadiran": kehadiran, "nilai_rata": nilai_rata,
            "jumlah_cuti": jumlah_cuti, "jumlah_mengulang": jumlah_mengulang,
            "lama_studi_semester": lama_studi, "status_kerja": status_kerja,
            "organisasi": org, "penghasilan_ortu": penghasilan, "jalur_masuk": jalur
        }])[FEATURES]

        model = model_lr if model_choice == "Logistic Regression" else model_rf
        pred  = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        if pred == 1:
            st.success("## ✅ LULUS TEPAT WAKTU")
            st.balloons()
        else:
            st.error("## ⚠️ BERPOTENSI TERLAMBAT")

        st.markdown("---")
        st.markdown("**Probabilitas:**")
        st.metric("✅ Tepat Waktu", f"{proba[1]*100:.1f}%")
        st.metric("⚠️ Terlambat",   f"{proba[0]*100:.1f}%")

        fig = go.Figure(go.Bar(
            x=["Tepat Waktu","Terlambat"],
            y=[proba[1]*100, proba[0]*100],
            marker_color=["#2E74B5","#E74C3C"],
            text=[f"{proba[1]*100:.1f}%", f"{proba[0]*100:.1f}%"],
            textposition="outside"
        ))
        fig.update_layout(yaxis_range=[0,110], yaxis_title="%",
                          title="Probabilitas Prediksi", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"*Model: **{model_choice}***")

        # Faktor risiko
        st.markdown("---")
        st.markdown("**⚠️ Faktor Risiko:**")
        risks = []
        if ipk < 2.5:          risks.append("IPK di bawah 2.5")
        if kehadiran < 75:     risks.append("Kehadiran < 75%")
        if jumlah_mengulang>3: risks.append("Banyak MK diulang")
        if status_kerja == 2:  risks.append("Bekerja full-time")
        if lama_studi > 9:     risks.append("Studi > 9 semester")
        if risks:
            for r in risks: st.warning(f"• {r}")
        else:
            st.info("✅ Tidak ada faktor risiko signifikan")
    else:
        st.info("👈 Isi data mahasiswa dan klik **Prediksi Sekarang**")
        st.markdown("""
        **Panduan Pengisian:**
        - IPK: skala 1.5 – 4.0
        - SKS: total SKS yang telah lulus
        - Kehadiran: persentase 0–100%
        - Nilai Rata-rata: 50–100
        """)