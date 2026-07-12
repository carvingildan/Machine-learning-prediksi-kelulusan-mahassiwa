import streamlit as st

st.set_page_config(page_title="Dokumentasi", page_icon="📋", layout="wide")

with st.sidebar:
    st.markdown("### 🎓 Prediksi Kelulusan")
    st.markdown("---")
    st.page_link("app.py",                    label="🏠 Beranda")
    st.page_link("pages/1_EDA.py",            label="📊 EDA & Analisis Data")
    st.page_link("pages/2_Prediksi.py",       label="🤖 Prediksi Mahasiswa")
    st.page_link("pages/3_Evaluasi_Model.py", label="📈 Evaluasi Model")
    st.page_link("pages/4_Interpretasi.py",   label="🔍 Interpretasi SHAP")
    st.page_link("pages/5_Dokumentasi.py",    label="📋 Dokumentasi")

st.title("📋 Dokumentasi Proyek")

tab1, tab2, tab3, tab4 = st.tabs(["📌 Dataset","⚙️ Metodologi","🏗️ Struktur Proyek","📚 Referensi"])

# ── TAB 1: Dataset ───────────────────────────────────────
with tab1:
    st.markdown("### 📌 Informasi Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Info | Detail |
        |------|--------|
        | **Nama Dataset** | Dataset Mahasiswa Sintetis |
        | **Jumlah Data** | 1.000 baris |
        | **Jumlah Fitur** | 11 fitur + 1 target |
        | **Tipe Data** | Numerik & Kategorik |
        | **Missing Value** | 0 |
        | **Target** | lulus_tepat_waktu (biner) |
        | **Distribusi Kelas** | 83% Tepat Waktu, 17% Terlambat |
        """)
    with col2:
        st.markdown("""
        **Sumber Data:**
        Dataset ini merupakan data sintetis yang dibangkitkan
        menggunakan distribusi statistik berbasis studi literatur
        mengenai faktor-faktor kelulusan mahasiswa di Indonesia.

        **Justifikasi:**
        - Melindungi privasi data mahasiswa riil
        - Distribusi dirancang mencerminkan kondisi nyata
        - Dapat diganti dengan data SIAK untuk implementasi produksi
        """)

    st.markdown("### 📊 Deskripsi Fitur")
    import pandas as pd
    fitur_df = pd.DataFrame([
        ["ipk","Float","1.5 – 4.0","Indeks Prestasi Kumulatif","Akademik"],
        ["sks","Integer","20 – 160","Jumlah SKS yang telah lulus","Akademik"],
        ["kehadiran","Float","40 – 100%","Persentase kehadiran kuliah","Akademik"],
        ["nilai_rata","Float","50 – 100","Rata-rata nilai ujian","Akademik"],
        ["jumlah_cuti","Integer","0 – 6","Jumlah semester cuti","Akademik"],
        ["jumlah_mengulang","Integer","0 – 10","Jumlah MK yang diulang","Akademik"],
        ["lama_studi_semester","Integer","6 – 14","Lama studi dalam semester","Akademik"],
        ["penghasilan_ortu","Kategorik","Rendah/Menengah/Tinggi","Tingkat penghasilan orang tua","Sosial"],
        ["status_kerja","Integer","0/1/2","Tidak Bekerja/Part-time/Full-time","Sosial"],
        ["jalur_masuk","Kategorik","SNBP/SNBT/Mandiri","Jalur penerimaan masuk","Administratif"],
        ["organisasi","Integer","0/1","Tidak Aktif / Aktif organisasi","Sosial"],
        ["lulus_tepat_waktu","Integer","0/1","TARGET: 1=Tepat Waktu, 0=Terlambat","Target"],
    ], columns=["Nama Fitur","Tipe","Range/Nilai","Deskripsi","Kategori"])
    st.dataframe(fitur_df, use_container_width=True, hide_index=True)

# ── TAB 2: Metodologi ────────────────────────────────────
with tab2:
    st.markdown("### ⚙️ Pipeline Machine Learning")
    st.markdown("""
    ```
    📦 Raw Data
        ↓
    🔍 EDA & Analisis Kualitas Data
        ↓
    ⚙️ Preprocessing
        ├── StandardScaler (fitur numerik)
        ├── OneHotEncoder (fitur kategorik)
        └── ColumnTransformer + Pipeline
        ↓
    ✂️ Train / Validation / Test Split (64% / 16% / 20%)
        ↓
    ⚖️ SMOTE (menyeimbangkan kelas minoritas)
        ↓
    🤖 Training Model
        ├── Logistic Regression
        └── Random Forest
        ↓
    🔧 Hyperparameter Tuning (GridSearchCV + 5-Fold CV)
        ↓
    📊 Evaluasi (Accuracy, Precision, Recall, F1, AUC)
        ↓
    🔍 Interpretasi SHAP
        ↓
    💾 Model Saving (joblib .pkl)
        ↓
    🚀 Deployment Streamlit
    ```
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🤖 Model 1: Logistic Regression")
        st.markdown("""
        - **Tipe:** Linear classifier
        - **Fungsi aktivasi:** Sigmoid
        - **Hyperparameter terbaik:**
          - C = 0.1
          - solver = lbfgs
          - penalty = l2
          - max_iter = 1000
        - **Kelebihan:** Cepat, interpretable, stabil
        - **Status:** ✅ **Model Terbaik (F1=0.9504)**
        """)
    with col2:
        st.markdown("#### 🌳 Model 2: Random Forest")
        st.markdown("""
        - **Tipe:** Ensemble (bagging)
        - **Jumlah pohon:** 200
        - **Hyperparameter terbaik:**
          - n_estimators = 200
          - max_depth = 8
          - min_samples_split = 2
        - **Kelebihan:** Robust, feature importance
        - **Status:** 🥈 Runner-up (F1=0.9213)
        """)

    st.markdown("#### 📊 Metrik Evaluasi yang Digunakan")
    metrik_df = pd.DataFrame([
        ["Accuracy","TP+TN / Total","Proporsi prediksi benar keseluruhan"],
        ["Precision","TP / (TP+FP)","Dari prediksi positif, berapa yang benar"],
        ["Recall","TP / (TP+FN)","Dari data positif aktual, berapa terdeteksi"],
        ["F1-Score","2×(P×R)/(P+R)","Harmonik Precision dan Recall"],
        ["ROC-AUC","Luas kurva ROC","Kemampuan model membedakan kelas"],
        ["Cross-Val 5-Fold","Rata-rata 5 lipatan","Stabilitas dan generalisasi model"],
    ], columns=["Metrik","Formula","Deskripsi"])
    st.dataframe(metrik_df, use_container_width=True, hide_index=True)

# ── TAB 3: Struktur Proyek ───────────────────────────────
with tab3:
    st.markdown("### 🏗️ Struktur Repository")
    st.code("""
capstone-prediksi-kelulusan/
│
├── data/
│   ├── raw/
│   │   └── dataset_mahasiswa.csv      # Data mentah
│   └── processed/
│       ├── X_test.csv                 # Data uji fitur
│       ├── y_test.csv                 # Label uji
│       ├── X_train.csv                # Data latih
│       └── model_results.csv          # Hasil evaluasi
│
├── models/
│   ├── best_model.pkl                 # Random Forest (tuned)
│   ├── logistic_regression.pkl        # Logistic Regression (tuned)
│   └── feature_info.pkl               # Info fitur
│
├── src/
│   ├── generate_data.py               # Generate dataset sintetis
│   └── train_model.py                 # Training & evaluasi model
│
├── app/
│   ├── app.py                         # Halaman utama Streamlit
│   └── pages/
│       ├── 1_EDA.py                   # Dashboard EDA interaktif
│       ├── 2_Prediksi.py              # Interface prediksi real-time
│       ├── 3_Evaluasi_Model.py        # Evaluasi & perbandingan model
│       ├── 4_Interpretasi.py          # SHAP interpretability
│       └── 5_Dokumentasi.py           # Halaman ini
│
├── reports/
│   └── laporan_prediksi_kelulusan.docx  # Laporan teknis lengkap
│
├── requirements.txt                   # Dependencies Python
└── README.md                          # Dokumentasi proyek
    """, language="")

    st.markdown("#### 🚀 Cara Menjalankan Aplikasi")
    st.code("""
# 1. Clone repository
git clone https://github.com/username/prediksi-kelulusan.git
cd prediksi-kelulusan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset dan train model (jika belum ada)
python src/generate_data.py
python src/train_model.py

# 4. Jalankan aplikasi Streamlit
cd app
streamlit run app.py
    """, language="bash")

# ── TAB 4: Referensi ─────────────────────────────────────
with tab4:
    st.markdown("### 📚 Referensi Ilmiah")
    refs = [
        ("Breiman (2001)", "Random forests. *Machine Learning*, 45(1), 5–32."),
        ("Chawla et al. (2002)", "SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357."),
        ("Cortes & Vapnik (1995)", "Support-vector networks. *Machine Learning*, 20(3), 273–297."),
        ("Géron (2019)", "*Hands-On ML with Scikit-Learn, Keras & TF* (2nd ed.). O'Reilly."),
        ("Lundberg & Lee (2017)", "A unified approach to interpreting model predictions. *NeurIPS*, 30."),
        ("Pedregosa et al. (2011)", "Scikit-learn: ML in Python. *JMLR*, 12, 2825–2830."),
        ("Romero & Ventura (2010)", "Educational data mining: A review. *IEEE TSMC*, 40(6), 601–618."),
        ("Quinlan (1986)", "Induction of decision trees. *Machine Learning*, 1(1), 81–106."),
    ]
    for author, desc in refs:
        st.markdown(f"- **{author}** — {desc}")

    st.markdown("---")
    st.markdown("### 🛠️ Teknologi yang Digunakan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📊 Data & ML**
        - Python 3.14
        - Pandas, NumPy
        - Scikit-learn
        - imbalanced-learn (SMOTE)
        """)
    with col2:
        st.markdown("""
        **📈 Visualisasi**
        - Plotly
        - Matplotlib
        - Seaborn
        - SHAP
        """)
    with col3:
        st.markdown("""
        **🚀 Deployment**
        - Streamlit
        - Joblib
        - GitHub
        - Streamlit Cloud
        """)