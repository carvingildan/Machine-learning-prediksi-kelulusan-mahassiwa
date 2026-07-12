import streamlit as st

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1F3864 0%,#2E74B5 100%);
padding:2rem;border-radius:12px;color:white;text-align:center;margin-bottom:2rem}
.nav-card{background:white;border-radius:12px;padding:1.5rem;
box-shadow:0 2px 15px rgba(0,0,0,.08);text-align:center;border-top:4px solid #2E74B5;height:100%}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎓 Prediksi Kelulusan")
    st.markdown("---")
    st.page_link("app.py",                       label="🏠 Beranda")
    st.page_link("pages/1_EDA.py",               label="📊 EDA & Analisis Data")
    st.page_link("pages/2_Prediksi.py",          label="🤖 Prediksi Mahasiswa")
    st.page_link("pages/3_Evaluasi_Model.py",    label="📈 Evaluasi Model")
    st.page_link("pages/4_Interpretasi.py",      label="🔍 Interpretasi SHAP")
    st.page_link("pages/5_Dokumentasi.py",       label="📋 Dokumentasi")
    st.markdown("---")
    st.info("👨‍💻 **Carvin Gildan**\nTeknik Informatika\nUDINUS 2025/2026")

st.markdown("""<div class="main-header">
<h1>🎓 Sistem Prediksi Kelulusan Mahasiswa</h1>
<p style="font-size:1.1rem;opacity:.9;margin:0">Machine Learning · Universitas Dian Nuswantoro · 2025/2026</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("📦 Total Data","1.000","mahasiswa")
c2.metric("🔢 Fitur","11","prediktor")
c3.metric("🎯 Accuracy","91.50%","model terbaik")
c4.metric("📊 F1-Score","0.9504","Logistic Regression")

st.markdown("---")
st.markdown("### 🗺️ Navigasi Aplikasi")
cols = st.columns(5)
items = [
    ("📊","EDA & Analisis","Eksplorasi distribusi, korelasi, dan visualisasi interaktif"),
    ("🤖","Prediksi","Input data mahasiswa dan dapatkan prediksi real-time"),
    ("📈","Evaluasi Model","Confusion matrix, ROC curve, perbandingan model"),
    ("🔍","Interpretasi SHAP","Analisis pengaruh setiap fitur terhadap prediksi"),
    ("📋","Dokumentasi","Dataset, metodologi, dan cara penggunaan aplikasi"),
]
for col,(icon,title,desc) in zip(cols,items):
    col.markdown(f'<div class="nav-card"><h2>{icon}</h2><h4>{title}</h4><p style="color:#666;font-size:.85rem">{desc}</p></div>', unsafe_allow_html=True)

st.markdown("---")
ca,cb = st.columns([3,2])
with ca:
    st.markdown("### 📌 Tentang Proyek")
    st.markdown("""Sistem ini memprediksi apakah mahasiswa akan **lulus tepat waktu** atau **terlambat**
berdasarkan 11 fitur akademik dan non-akademik menggunakan Machine Learning.

**Pipeline lengkap:**
- ✅ Data Acquisition & Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering (Scaling + OneHotEncoding)
- ✅ Training: Logistic Regression & Random Forest
- ✅ Hyperparameter Tuning (GridSearchCV)
- ✅ SMOTE untuk imbalanced data
- ✅ SHAP untuk interpretabilitas model
- ✅ Deployment via Streamlit""")
with cb:
    st.markdown("### 🏆 Hasil Model Terbaik")
    st.markdown("""| Metrik | Nilai |\n|--------|-------|\n|**Accuracy**|91.50%|\n|**Precision**|92.09%|\n|**Recall**|98.19%|\n|**F1-Score**|0.9504|\n|**ROC-AUC**|0.9438|""")
    st.success("🥇 Model: **Logistic Regression**")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:.85rem">🎓 UAS Pembelajaran Mesin 2025/2026 · UDINUS · <b>Carvin Gildan</b></div>', unsafe_allow_html=True)