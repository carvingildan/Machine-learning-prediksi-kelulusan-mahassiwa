import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, classification_report)


def find_path(filenames):
    prefixes = ["", "../", "../../"]
    for name in filenames:
        for prefix in prefixes:
            full = prefix + name
            if os.path.exists(full):
                return full
    return None

st.set_page_config(page_title="Evaluasi Model", page_icon="📈", layout="wide")

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
def load_all():
    lr_path    = find_path(["models/logistic_regression.pkl"])
    rf_path    = find_path(["models/best_model.pkl"])
    xtest_path = find_path(["data/processed/X_test.csv"])
    ytest_path = find_path(["data/processed/y_test.csv"])
    info_path  = find_path(["models/feature_info.pkl"])
    if not all([lr_path, rf_path, xtest_path, ytest_path, info_path]):
        st.error("❌ Model/data tidak ditemukan! Jalankan src/main.py dulu.")
        st.stop()
    lr   = joblib.load(lr_path)
    rf   = joblib.load(rf_path)
    X    = pd.read_csv(xtest_path)
    y    = pd.read_csv(ytest_path).squeeze()
    info = joblib.load(info_path)
    return lr, rf, X, y, info

model_lr, model_rf, X_test, y_test, feat_info = load_all()
FEATURES = feat_info["num"] + feat_info["cat"]

st.title("📈 Evaluasi & Perbandingan Model")

# Hitung metrik
def get_metrics(model, X, y, name):
    yp = model.predict(X); ypr = model.predict_proba(X)[:,1]
    return {"Model":name, "Accuracy":accuracy_score(y,yp),
            "Precision":precision_score(y,yp), "Recall":recall_score(y,yp),
            "F1-Score":f1_score(y,yp), "ROC-AUC":roc_auc_score(y,ypr)}

m1 = get_metrics(model_lr, X_test[FEATURES], y_test, "Logistic Regression")
m2 = get_metrics(model_rf, X_test[FEATURES], y_test, "Random Forest")
df_m = pd.DataFrame([m1,m2])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Perbandingan","🔲 Confusion Matrix","📉 ROC Curve","📋 Classification Report"])

# ── TAB 1 ────────────────────────────────────────────────
with tab1:
    st.markdown("#### 🏆 Perbandingan Semua Model")
    st.dataframe(df_m.style.format({c:"{:.4f}" for c in df_m.columns if c!="Model"})
                 .highlight_max(subset=[c for c in df_m.columns if c!="Model"], color="#d4edda"), use_container_width=True)

    metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    fig = go.Figure()
    colors = ["#2E74B5","#E74C3C"]
    for i, row in df_m.iterrows():
        fig.add_trace(go.Bar(name=row["Model"], x=metrics, y=[row[m] for m in metrics],
                             marker_color=colors[i], text=[f"{row[m]:.4f}" for m in metrics],
                             textposition="outside"))
    fig.update_layout(barmode="group", yaxis_range=[0.7,1.05],
                      title="Perbandingan Metrik Evaluasi", height=420,
                      yaxis_title="Skor")
    st.plotly_chart(fig, use_container_width=True)

    best_model = "Logistic Regression" if m1["F1-Score"] >= m2["F1-Score"] else "Random Forest"
    st.success(f"🥇 **Model Terbaik: {best_model}** (berdasarkan F1-Score tertinggi)")

# ── TAB 2 ────────────────────────────────────────────────
with tab2:
    st.markdown("#### 🔲 Confusion Matrix")
    model_sel = st.radio("Pilih Model:", ["Logistic Regression","Random Forest"], horizontal=True)
    model = model_lr if model_sel=="Logistic Regression" else model_rf

    yp = model.predict(X_test[FEATURES])
    cm = confusion_matrix(y_test, yp)
    labels = ["Terlambat","Tepat Waktu"]

    fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                    color_continuous_scale="Blues",
                    title=f"Confusion Matrix — {model_sel}",
                    labels=dict(x="Prediksi",y="Aktual"))
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    TP,FP = cm[1][1], cm[0][1]
    TN,FN = cm[0][0], cm[1][0]
    c1.metric("✅ True Positive",  TP)
    c2.metric("✅ True Negative",  TN)
    c3.metric("❌ False Positive", FP)
    c4.metric("❌ False Negative", FN)

# ── TAB 3 ────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📉 ROC Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                             line=dict(dash="dash", color="gray")))
    for model, name, color in [(model_lr,"Logistic Regression","#2E74B5"),
                                (model_rf,"Random Forest","#E74C3C")]:
        ypr = model.predict_proba(X_test[FEATURES])[:,1]
        fpr, tpr, _ = roc_curve(y_test, ypr)
        auc = roc_auc_score(y_test, ypr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.4f})", line=dict(color=color, width=2)))
    fig.update_layout(title="ROC Curve — Perbandingan Model", height=480,
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 4 ────────────────────────────────────────────────
with tab4:
    st.markdown("#### 📋 Classification Report")
    model_sel2 = st.radio("Model:", ["Logistic Regression","Random Forest"], horizontal=True, key="cr")
    model2 = model_lr if model_sel2=="Logistic Regression" else model_rf
    yp2 = model2.predict(X_test[FEATURES])
    report = classification_report(y_test, yp2, target_names=["Terlambat","Tepat Waktu"], output_dict=True)
    df_rep = pd.DataFrame(report).T.round(4)
    st.dataframe(df_rep.style.highlight_max(color="#d4edda"), use_container_width=True)
    st.code(classification_report(y_test, yp2, target_names=["Terlambat","Tepat Waktu"]))