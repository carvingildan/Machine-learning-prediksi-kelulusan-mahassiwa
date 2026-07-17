# 🎓 Sistem Prediksi Kelulusan Mahasiswa

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9-orange?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3-green?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-0.52-purple?style=for-the-badge)

**Capstone Project — UAS Pembelajaran Mesin 2025/2026**
**Mahasiswa:** Carvin Gildan Sirajudin
**NIM:** A11.2024.15822
Universitas Dian Nuswantoro · Teknik Informatika

🚀 **[Live Demo →](https://machine-learning-prediksi-kelulusan-mahassiwa-7wqhw5jgxvengfz6.streamlit.app/)**

</div>

---

## 📌 Deskripsi Proyek

Sistem prediksi kelulusan mahasiswa berbasis **Machine Learning** yang memprediksi apakah seorang mahasiswa akan **lulus tepat waktu** atau **terlambat** berdasarkan data akademik dan non-akademik.

Proyek ini merupakan implementasi **end-to-end pipeline Machine Learning** mulai dari:
- 📦 Akuisisi & pembuatan dataset
- 🔍 Exploratory Data Analysis (EDA)
- ⚙️ Preprocessing & Feature Engineering
- 🤖 Training 3 Model ML (LR, RF, XGBoost)
- 🔧 Hyperparameter Tuning (GridSearchCV)
- ⚖️ Penanganan Imbalanced Data (SMOTE)
- 🔎 Interpretasi Model (SHAP)
- 🚀 Deployment via Streamlit Community Cloud

---

## 🏆 Hasil Model Terbaik

| Metrik | Logistic Regression | Random Forest | XGBoost |
|--------|-------------------|---------------|---------|
| **Accuracy** | **91.50%** | 86.50% | 88.00% |
| **Precision** | **92.09%** | 86.01% | 88.50% |
| **Recall** | **98.19%** | 100.00% | 97.00% |
| **F1-Score** | **0.9504** | 0.9213 | 0.9255 |
| **ROC-AUC** | **0.9438** | 0.9134 | 0.9280 |

🥇 **Model Terpilih: Logistic Regression** (berdasarkan F1-Score tertinggi)

---

## 🗂️ Struktur Repository

```
Machine-learning-prediksi-kelulusan-mahassiwa/
│
├── app/                              # Aplikasi Streamlit
│   ├── app.py                        # Halaman utama (Beranda)
│   └── pages/
│       ├── 1_EDA.py                  # Dashboard EDA interaktif
│       ├── 2_Prediksi.py             # Interface prediksi real-time
│       ├── 3_Evaluasi_Model.py       # Evaluasi & perbandingan model
│       ├── 4_Interpretasi.py         # Interpretasi SHAP
│       └── 5_Dokumentasi.py          # Dokumentasi proyek
│
├── data/
│   ├── raw/
│   │   └── dataset_mahasiswa.csv     # Dataset mentah (1.000 baris)
│   └── processed/                    # Data hasil preprocessing
│       ├── X_test.csv
│       ├── y_test.csv
│       ├── X_train.csv
│       └── model_results.csv
│
├── models/                           # Model tersimpan
│   ├── logistic_regression.pkl       # Logistic Regression (Best)
│   ├── best_model.pkl                # Random Forest
│   ├── xgboost_model.pkl             # XGBoost
│   └── feature_info.pkl              # Informasi fitur
│
├── notebooks/                        # Jupyter Notebooks
│   ├── 01_eda.ipynb                  # EDA & analisis data
│   └── 02_modeling.ipynb             # Modeling, tuning & SHAP
│
├── src/                              # Script Python
│   ├── generate_dataset.py           # Generate dataset sintetis
│   └── main.py                       # Training 3 model + evaluasi
│
├── output/                           # Hasil visualisasi
│   ├── evaluasi_model.png
│   ├── hasil.txt
│   └── ...
│
├── reports/
│   └── laporan_final_uas.docx        # Laporan teknis lengkap
│
├── requirements.txt                  # Dependencies Python
├── runtime.txt                       # Python runtime untuk Streamlit Cloud
├── README.md                         # File ini
└── .gitignore
```

---

## 🖥️ Fitur Aplikasi Streamlit

| Halaman | Deskripsi |
|---------|-----------|
| 🏠 **Beranda** | Overview proyek, statistik ringkas (accuracy, F1), navigasi |
| 📊 **EDA & Analisis** | 5 tab: Overview, Distribusi, Korelasi, Boxplot, Kategorik — semua interaktif |
| 🤖 **Prediksi** | Input data mahasiswa → prediksi real-time + probabilitas + faktor risiko |
| 📈 **Evaluasi Model** | Confusion matrix, ROC curve, classification report per model |
| 🔍 **Interpretasi SHAP** | Feature importance, beeswarm plot, waterfall chart individual |
| 📋 **Dokumentasi** | Info dataset, pipeline, struktur repo, cara penggunaan, referensi |

---

## 📊 Dataset

| Info | Detail |
|------|--------|
| **Jumlah Data** | 1.000 mahasiswa |
| **Jumlah Fitur** | 11 fitur + 1 target |
| **Tipe Data** | Numerik & Kategorik |
| **Target** | lulus_tepat_waktu (0=Terlambat, 1=Tepat Waktu) |
| **Distribusi Kelas** | 83% Tepat Waktu, 17% Terlambat |
| **Jenis Dataset** | Sintetis (dapat diganti data SIAK riil) |

### Fitur yang Digunakan:
| No | Fitur | Tipe | Keterangan |
|----|-------|------|------------|
| 1 | ipk | Float | Indeks Prestasi Kumulatif (1.5–4.0) |
| 2 | sks | Integer | Jumlah SKS yang telah lulus |
| 3 | kehadiran | Float | Persentase kehadiran kuliah |
| 4 | nilai_rata | Float | Rata-rata nilai ujian |
| 5 | jumlah_cuti | Integer | Jumlah semester cuti |
| 6 | jumlah_mengulang | Integer | Jumlah MK yang diulang |
| 7 | lama_studi_semester | Integer | Lama studi dalam semester |
| 8 | penghasilan_ortu | Kategorik | Rendah / Menengah / Tinggi |
| 9 | status_kerja | Integer | 0=Tidak, 1=Part-time, 2=Full-time |
| 10 | jalur_masuk | Kategorik | SNBP / SNBT / Mandiri |
| 11 | organisasi | Integer | 0=Tidak Aktif, 1=Aktif |

---

## 🤖 Pipeline Machine Learning

```
📦 Dataset (1.000 mahasiswa)
    ↓
🔍 EDA & Analisis Kualitas Data
    ↓
⚙️ Preprocessing
    ├── StandardScaler  → fitur numerik
    └── OneHotEncoder   → fitur kategorik
    ↓
✂️ Train / Validation / Test Split (64% / 16% / 20%)
    ↓
⚖️ SMOTE → menyeimbangkan kelas (83:17 → 50:50)
    ↓
🤖 Training 3 Model
    ├── Logistic Regression
    ├── Random Forest
    └── XGBoost
    ↓
🔧 Hyperparameter Tuning (GridSearchCV + 5-Fold CV)
    ↓
📊 Evaluasi (Accuracy, Precision, Recall, F1, AUC)
    ↓
🔎 Interpretasi SHAP (TreeExplainer)
    ↓
💾 Simpan Model (.pkl)
    ↓
🚀 Deployment Streamlit Cloud
```

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone Repository
```bash
git clone https://github.com/carvingildan/Machine-learning-prediksi-kelulusan-mahassiwa.git
cd Machine-learning-prediksi-kelulusan-mahassiwa
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Model
```bash
cd src
python generate_dataset.py
python main.py
cd ..
```

### 4. Jalankan Streamlit
```bash
cd app
python -m streamlit run app.py
```

Buka browser: **http://localhost:8501**

### 5. Jalankan Notebook (Opsional)
```bash
pip install jupyter
cd notebooks
jupyter notebook
```

---

## 🛠️ Teknologi yang Digunakan

| Kategori | Library |
|----------|---------|
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Model Interpretation** | SHAP |
| **Visualisasi** | Plotly, Matplotlib, Seaborn |
| **Deployment** | Streamlit |
| **Model Persistence** | Joblib |
| **Notebook** | Jupyter |

---

## 📚 Referensi

- Breiman (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen & Guestrin (2016). XGBoost. *KDD 2016*, 785–794.
- Chawla et al. (2002). SMOTE. *JAIR*, 16, 321–357.
- Lundberg & Lee (2017). SHAP. *NeurIPS*, 30.
- Pedregosa et al. (2011). Scikit-learn. *JMLR*, 12, 2825–2830.
- Romero & Ventura (2010). Educational data mining. *IEEE TSMC*, 40(6).

---

## 👨‍💻 Author

**Carvin Gildan Sirajudin**
    **A11.2024.15822**
Program Studi Teknik Informatika
Universitas Dian Nuswantoro
UAS Pembelajaran Mesin 2025/2026

---

<div align="center">

⭐ Jika proyek ini bermanfaat, silakan beri bintang di GitHub!

🚀 **[Coba Aplikasi Live →](https://machine-learning-prediksi-kelulusan-mahassiwa-xscxhsnawej3hssl.streamlit.app)**

</div>