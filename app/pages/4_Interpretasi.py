import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Interpretasi SHAP", page_icon="🔍", layout="wide")

with st.sidebar:
    st.markdown("### 🎓 Prediksi Kelulusan")
    st.markdown("---")
    st.page_link("app.py",                    label="🏠 Beranda")
    st.page_link("pages/1_EDA.py",            label="📊 EDA & Analisis Data")
    st.page_link("pages/2_Prediksi.py",       label="🤖 Prediksi Mahasiswa")
    st.page_link("pages/3_Evaluasi_Model.py", label="📈 Evaluasi Model")
    st.page_link("pages/4_Interpretasi.py",   label="🔍 Interpretasi SHAP")
    st.page_link("pages/5_Dokumentasi.py",    label="📋 Dokumentasi")

def find_path(filenames):
    prefixes = ["", "../", "../../", "/mount/src/machine-learning-prediksi-kelulusan-mahassiwa/"]
    for name in filenames:
        for prefix in prefixes:
            full = prefix + name
            if os.path.exists(full):
                return full
    return None

@st.cache_resource
def load_all():
    model_path = find_path(["models/best_model.pkl"])
    info_path  = find_path(["models/feature_info.pkl"])
    xtest_path = find_path(["data/processed/X_test.csv"])
    if not model_path:
        st.error("❌ Model tidak ditemukan!")
        st.stop()
    import joblib
    rf   = joblib.load(model_path)
    info = joblib.load(info_path)
    X    = pd.read_csv(xtest_path)
    return rf, info, X

model_rf, feat_info, X_test = load_all()
NUM = feat_info["num"]
CAT = feat_info["cat"]
FEATURES = NUM + CAT

st.title("🔍 Interpretasi Model dengan SHAP")
st.markdown("SHAP (SHapley Additive exPlanations) menjelaskan **pengaruh setiap fitur** terhadap prediksi model.")

@st.cache_data
def get_shap_values():
    import shap
    prep     = model_rf.named_steps["prep"]
    rf_model = model_rf.named_steps["model"]
    X_tr     = prep.transform(X_test[FEATURES])
    ohe_cols = prep.named_transformers_["cat"].get_feature_names_out(CAT)
    all_cols = NUM + list(ohe_cols)
    X_df     = pd.DataFrame(X_tr, columns=all_cols)

    explainer = shap.TreeExplainer(rf_model)

    # ── Fix: gunakan check_additivity=False untuk kompatibilitas SHAP terbaru ──
    try:
        shap_values = explainer.shap_values(X_df, check_additivity=False)
    except TypeError:
        shap_values = explainer.shap_values(X_df)

    # Handle berbagai format output SHAP
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])
    elif hasattr(shap_values, 'values'):
        sv = shap_values.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    return sv, X_df, all_cols

with st.spinner("⏳ Menghitung SHAP values..."):
    try:
        shap_vals, X_proc, all_cols = get_shap_values()
        shap_ok = True
    except Exception as e:
        shap_ok = False
        shap_error = str(e)

if not shap_ok:
    st.warning(f"⚠️ SHAP tidak dapat dihitung: {shap_error}")
    st.info("Menampilkan Feature Importance dari Random Forest sebagai alternatif.")

    # Fallback: tampilkan feature importance dari model langsung
    prep     = model_rf.named_steps["prep"]
    rf_model = model_rf.named_steps["model"]
    ohe_cols = prep.named_transformers_["cat"].get_feature_names_out(CAT)
    all_cols = NUM + list(ohe_cols)
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({"Fitur": all_cols, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)

    fig = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Fitur"],
        orientation="h", marker_color="#2E74B5",
        text=fi_df["Importance"].round(4), textposition="outside"
    ))
    fig.update_layout(title="Feature Importance — Random Forest", height=480,
                      xaxis_title="Importance Score")
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Feature Importance","🌊 SHAP Beeswarm","🔎 Prediksi Individual"])

# ── TAB 1: Feature Importance SHAP ───────────────────────
with tab1:
    st.markdown("#### 📊 SHAP Feature Importance (Mean |SHAP value|)")
    mean_shap = np.abs(shap_vals).mean(axis=0)
    if mean_shap.ndim > 1:
        mean_shap = mean_shap.flatten()

    fi_df = pd.DataFrame({"Fitur": all_cols, "SHAP Importance": mean_shap})
    fi_df = fi_df.sort_values("SHAP Importance", ascending=True).tail(12)

    colors = ["#E74C3C" if v >= fi_df["SHAP Importance"].quantile(0.75) else "#2E74B5"
              for v in fi_df["SHAP Importance"]]
    fig = go.Figure(go.Bar(
        x=fi_df["SHAP Importance"], y=fi_df["Fitur"],
        orientation="h", marker_color=colors,
        text=fi_df["SHAP Importance"].round(4), textposition="outside"
    ))
    fig.update_layout(title="Top 12 Fitur Terpenting (SHAP)", height=480,
                      xaxis_title="Mean |SHAP Value|")
    st.plotly_chart(fig, use_container_width=True)

    fi_show = fi_df.sort_values("SHAP Importance", ascending=False).copy()
    fi_show["Rank"] = range(1, len(fi_show)+1)
    fi_show["SHAP Importance"] = fi_show["SHAP Importance"].round(4)
    fi_show["Pengaruh"] = fi_show["SHAP Importance"].apply(
        lambda x: "🔴 Sangat Tinggi" if x > 0.12 else ("🟡 Tinggi" if x > 0.07 else "🟢 Sedang"))
    st.dataframe(fi_show[["Rank","Fitur","SHAP Importance","Pengaruh"]].reset_index(drop=True),
                 use_container_width=True)

# ── TAB 2: Beeswarm ──────────────────────────────────────
with tab2:
    st.markdown("#### 🌊 SHAP Beeswarm Plot")
    st.markdown("Setiap titik = satu mahasiswa. **Merah** = nilai fitur tinggi, **Biru** = rendah.")

    import shap
    top_n = st.slider("Top N fitur:", 5, min(15, len(all_cols)), 10)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.flatten()
    top_idx  = np.argsort(mean_abs)[-top_n:]
    sv_top   = shap_vals[:, top_idx]
    X_top    = X_proc.iloc[:, top_idx]
    cols_top = [all_cols[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, top_n*0.6 + 1))
    shap.summary_plot(sv_top, X_top, feature_names=cols_top, show=False, plot_size=None)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.info("""
    **Interpretasi:**
    - IPK tinggi → prediksi Tepat Waktu (SHAP positif)
    - Jumlah mengulang tinggi → prediksi Terlambat (SHAP negatif)
    - Kehadiran tinggi → prediksi Tepat Waktu
    - Lama studi panjang → prediksi Terlambat
    """)

# ── TAB 3: Individual ────────────────────────────────────
with tab3:
    st.markdown("#### 🔎 Analisis SHAP Prediksi Individual")
    sample_idx = st.slider("Pilih mahasiswa ke-:", 0, len(X_test)-1, 0)

    row = X_test[FEATURES].iloc[[sample_idx]]
    prep     = model_rf.named_steps["prep"]
    rf_model = model_rf.named_steps["model"]
    X_tr_row = prep.transform(row)
    ohe_cols = prep.named_transformers_["cat"].get_feature_names_out(CAT)
    all_cols_full = NUM + list(ohe_cols)
    X_df_row = pd.DataFrame(X_tr_row, columns=all_cols_full)

    import shap
    explainer = shap.TreeExplainer(rf_model)
    try:
        sv_row = explainer.shap_values(X_df_row, check_additivity=False)
    except TypeError:
        sv_row = explainer.shap_values(X_df_row)

    if isinstance(sv_row, list):
        sv_single = np.array(sv_row[1]).flatten()
    elif hasattr(sv_row, 'values'):
        sv_arr = sv_row.values
        sv_single = sv_arr[0,:,1] if sv_arr.ndim==3 else sv_arr[0]
    else:
        sv_arr = np.array(sv_row)
        sv_single = sv_arr[0,:,1] if sv_arr.ndim==3 else sv_arr[0]

    pred  = model_rf.predict(row)[0]
    proba = model_rf.predict_proba(row)[0]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Data Mahasiswa:**")
        st.dataframe(row.T.rename(columns={row.index[0]:"Nilai"}), use_container_width=True)
        if pred == 1:
            st.success(f"✅ Prediksi: **Tepat Waktu** ({proba[1]*100:.1f}%)")
        else:
            st.error(f"⚠️ Prediksi: **Terlambat** ({proba[0]*100:.1f}%)")

    with c2:
        st.markdown("**SHAP Waterfall — Kontribusi Tiap Fitur:**")
        top_k   = min(10, len(sv_single))
        idx_top = np.argsort(np.abs(sv_single))[-top_k:]
        sv_top_s   = sv_single[idx_top]
        cols_top_s = [all_cols_full[i] for i in idx_top]

        fig = go.Figure(go.Waterfall(
            name="SHAP", orientation="h",
            measure=["relative"]*top_k,
            x=list(sv_top_s[::-1]),
            y=list(cols_top_s[::-1]),
            connector={"line":{"color":"gray","width":1}},
            increasing={"marker":{"color":"#2E74B5"}},
            decreasing={"marker":{"color":"#E74C3C"}},
        ))
        fig.update_layout(title=f"SHAP Waterfall — Mahasiswa #{sample_idx}",
                          height=380, xaxis_title="SHAP Value")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔵 Positif = mendorong Tepat Waktu | 🔴 Negatif = mendorong Terlambat")