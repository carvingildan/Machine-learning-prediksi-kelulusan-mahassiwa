# 🎓 Prediksi Kelulusan Mahasiswa — Capstone Project ML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Capstone Project — UAS Pembelajaran Mesin 2025/2026**  
Universitas Dian Nuswantoro · Teknik Informatika  
**Mahasiswa:** Carvin Gildan

</div>

---

## 📌 Deskripsi Proyek

Sistem prediksi kelulusan mahasiswa berbasis **Machine Learning** yang memprediksi apakah seorang mahasiswa akan **lulus tepat waktu** atau **terlambat** berdasarkan data akademik dan non-akademik.

Proyek ini merupakan implementasi end-to-end pipeline Machine Learning mulai dari akuisisi data, eksplorasi data (EDA), preprocessing, pemodelan, evaluasi, interpretasi model dengan SHAP, hingga deployment menggunakan Streamlit.

### 🎯 Problem Statement
Keterlambatan kelulusan mahasiswa merupakan masalah serius yang berdampak pada akreditasi institusi dan masa depan mahasiswa. Dengan memanfaatkan data akademik (IPK, kehadiran, nilai rata-rata) dan non-akademik (status kerja, penghasilan orang tua, keaktifan organisasi), sistem ini dapat mengidentifikasi mahasiswa berisiko sejak dini sehingga intervensi dapat dilakukan tepat waktu.

---

## 🏗️ Struktur Repository

```
prediksi-kelulusan-mahasiswa/
│
├── app/                          # Aplikasi Streamlit
│   ├── app.py                    # Halaman utama
│   └── pages/
│       ├── 1_EDA.py              # Dashboard EDA interaktif
│       ├── 2_Prediksi.py         # Interface prediksi real-time
│       ├── 3_Evaluasi_Model.py   # Evaluasi & perbandingan model
│       ├── 4_Interpretasi.py     # Interpretasi SHAP
│       └── 5_Dokumentasi.py      # Dokumentasi proyek
│
├── data/
│   ├── raw/
│   │   └── dataset_mahasiswa.csv # Dataset mentah
│   └── processed/                # Data hasil preprocessing
│       ├── X_test.csv
│       ├── y_test.csv
│       └── X_train.csv
│
├── models/                       # Model tersimpan
│   ├── best_model.pkl            # Random Forest (tuned)
│   ├── logistic_regression.pkl   # Logistic Regression
│   └── feature_info.pkl          # Informasi fitur
│
├── src/                          # Script Python
│   ├── generate_dataset.py       # Generate dataset sintetis
│   └── main.py                   # Training & evaluasi model
│
├── reports/
│   └── laporan_prediksi_kelulusan.docx  # Laporan teknis
│
├── requirements.txt              # Dependencies
└── README.md                     # File ini
```

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone Repository
```bash
git clone https://github.com/USERNAME/prediksi-kelulusan-mahasiswa.git
cd prediksi-kelulusan-mahasiswa
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

### 4. Jalankan Aplikasi Streamlit
```bash
cd app
python -m streamlit run app.py
```

Buka browser: **http://localhost:8501**

---

## 🖥️ Fitur Aplikasi

| Halaman | Deskripsi |
|---------|-----------|
| 🏠 **Beranda** | Overview proyek, statistik ringkas, navigasi |
| 📊 **EDA** | Distribusi data, korelasi, boxplot, analisis kategorik interaktif |
| 🤖 **Prediksi** | Input data mahasiswa → prediksi real-time + faktor risiko |
| 📈 **Evaluasi Model** | Confusion matrix, ROC curve, classification report |
| 🔍 **Interpretasi SHAP** | Feature importance, beeswarm plot, waterfall individual |
| 📋 **Dokumentasi** | Dataset, metodologi, struktur proyek, referensi |

---

## 📊 Dataset

| Info | Detail |
|------|--------|
| Jumlah Data | 1.000 mahasiswa |
| Jumlah Fitur | 11 fitur + 1 target |
| Tipe | Numerik & Kategorik |
| Target | lulus_tepat_waktu (0/1) |
| Distribusi | 83% Tepat Waktu, 17% Terlambat |

### Fitur yang Digunakan:
- **Akademik:** IPK, SKS, Kehadiran, Nilai Rata-rata, Jumlah Cuti, Jumlah Mengulang, Lama Studi
- **Sosial:** Penghasilan Orang Tua, Status Kerja, Jalur Masuk, Keaktifan Organisasi

---

## 🤖 Model Machine Learning

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **Logistic Regression** ⭐ | **91.50%** | **0.9504** | **0.9438** |
| Random Forest | 86.50% | 0.9213 | 0.9243 |

### Teknik yang Digunakan:
- ✅ StandardScaler + OneHotEncoder (preprocessing)
- ✅ SMOTE (penanganan imbalanced data)
- ✅ GridSearchCV + 5-Fold Cross Validation (hyperparameter tuning)
- ✅ SHAP (interpretabilitas model)

---

## 🛠️ Teknologi

```
Python 3.14        Pandas          NumPy
Scikit-learn       Streamlit       Plotly
Matplotlib         Seaborn         SHAP
imbalanced-learn   Joblib
```

---

## 📚 Referensi

- Breiman (2001). Random forests. *Machine Learning*, 45(1), 5–32.
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
