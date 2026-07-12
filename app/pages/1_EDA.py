import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

with st.sidebar:
    st.markdown("### 🎓 Prediksi Kelulusan")
    st.markdown("---")
    st.page_link("app.py",                    label="🏠 Beranda")
    st.page_link("pages/1_EDA.py",            label="📊 EDA & Analisis Data")
    st.page_link("pages/2_Prediksi.py",       label="🤖 Prediksi Mahasiswa")
    st.page_link("pages/3_Evaluasi_Model.py", label="📈 Evaluasi Model")
    st.page_link("pages/4_Interpretasi.py",   label="🔍 Interpretasi SHAP")
    st.page_link("pages/5_Dokumentasi.py",    label="📋 Dokumentasi")

@st.cache_data
def load_data():
    paths = [
        "data/raw/dataset_mahasiswa.csv",
        "../data/raw/dataset_mahasiswa.csv",
        "data/dataset_mahasiswa.csv",
        "../data/dataset_mahasiswa.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("❌ File dataset tidak ditemukan!")
    st.stop()

df = load_data()
df["Status"] = df["lulus_tepat_waktu"].map({1:"Tepat Waktu", 0:"Terlambat"})

st.title("📊 Exploratory Data Analysis")
st.markdown("Analisis mendalam terhadap dataset mahasiswa untuk memahami pola dan distribusi data.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Overview","📊 Distribusi","🔗 Korelasi","📦 Boxplot","🏷️ Kategorik"])

with tab1:
    st.markdown("#### 📋 Informasi Dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Baris", f"{len(df):,}")
    c2.metric("Total Fitur", df.shape[1]-2)
    c3.metric("Missing Value", df.isnull().sum().sum())
    c4.metric("Duplikat", df.duplicated().sum())

    counts = df["Status"].value_counts()
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index,
                     color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"},
                     title="Proporsi Kelas Target")
        fig.update_traces(textinfo="percent+label+value")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.bar(x=counts.index, y=counts.values, color=counts.index,
                      color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"},
                      title="Jumlah per Kelas", text=counts.values)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    NUM = ["ipk","sks","kehadiran","nilai_rata","jumlah_cuti","jumlah_mengulang","lama_studi_semester"]
    st.markdown("#### 📊 Statistik Deskriptif")
    st.dataframe(df[NUM].describe().round(2).T, use_container_width=True)
    st.markdown("#### 🔍 Preview Dataset")
    st.dataframe(df.drop(columns=["Status"]).head(10), use_container_width=True)

with tab2:
    NUM = ["ipk","sks","kehadiran","nilai_rata","jumlah_cuti","jumlah_mengulang","lama_studi_semester"]
    fitur_sel = st.selectbox("Pilih fitur:", NUM)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=fitur_sel, color="Status", barmode="overlay",
                           color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"},
                           title=f"Distribusi {fitur_sel}", opacity=0.7, nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.violin(df, y=fitur_sel, x="Status", color="Status", box=True,
                         color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"},
                         title=f"Violin Plot {fitur_sel}")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    NUM_TARGET = ["ipk","sks","kehadiran","nilai_rata","jumlah_cuti","jumlah_mengulang","lama_studi_semester","lulus_tepat_waktu"]
    corr = df[NUM_TARGET].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdYlGn",
                    zmin=-1, zmax=1, title="Heatmap Korelasi", aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    corr_target = df[NUM_TARGET].corr()["lulus_tepat_waktu"].drop("lulus_tepat_waktu").sort_values()
    colors_bar = ["#E74C3C" if v < 0 else "#2E74B5" for v in corr_target]
    fig2 = go.Figure(go.Bar(x=corr_target.values, y=corr_target.index,
                            orientation="h", marker_color=colors_bar))
    fig2.update_layout(title="Korelasi vs Target", height=350)
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    NUM = ["ipk","sks","kehadiran","nilai_rata","jumlah_cuti","jumlah_mengulang","lama_studi_semester"]
    fig = make_subplots(rows=2, cols=4, subplot_titles=NUM)
    for i, col in enumerate(NUM):
        r, c = divmod(i, 4)
        for status, color in {"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"}.items():
            fig.add_trace(go.Box(y=df[df["Status"]==status][col], name=status,
                                 marker_color=color, showlegend=(i==0)), row=r+1, col=c+1)
    fig.update_layout(boxmode="group", height=500, title_text="Boxplot per Kelas")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    col1, col2 = st.columns(2)
    with col1:
        ct = pd.crosstab(df["penghasilan_ortu"], df["Status"])
        ct_pct = ct.div(ct.sum(axis=1), axis=0)*100
        fig = px.bar(ct_pct.reset_index(), x="penghasilan_ortu",
                     y=["Tepat Waktu","Terlambat"], barmode="stack",
                     title="Penghasilan Ortu vs Status (%)",
                     color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ct2 = pd.crosstab(df["jalur_masuk"], df["Status"])
        ct2_pct = ct2.div(ct2.sum(axis=1), axis=0)*100
        fig2 = px.bar(ct2_pct.reset_index(), x="jalur_masuk",
                      y=["Tepat Waktu","Terlambat"], barmode="stack",
                      title="Jalur Masuk vs Status (%)",
                      color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"})
        st.plotly_chart(fig2, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        org = df.groupby(["organisasi","Status"]).size().reset_index(name="n")
        org["organisasi"] = org["organisasi"].map({0:"Tidak Aktif",1:"Aktif"})
        fig3 = px.bar(org, x="organisasi", y="n", color="Status", barmode="group",
                      title="Organisasi vs Kelulusan",
                      color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"})
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        kerja = df.groupby(["status_kerja","Status"]).size().reset_index(name="n")
        kerja["status_kerja"] = kerja["status_kerja"].map({0:"Tidak Bekerja",1:"Part-time",2:"Full-time"})
        fig4 = px.bar(kerja, x="status_kerja", y="n", color="Status", barmode="group",
                      title="Status Kerja vs Kelulusan",
                      color_discrete_map={"Tepat Waktu":"#2E74B5","Terlambat":"#E74C3C"})
        st.plotly_chart(fig4, use_container_width=True)