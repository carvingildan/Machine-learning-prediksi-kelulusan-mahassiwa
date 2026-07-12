"""
main.py  ──  Sistem Prediksi Kelulusan Mahasiswa
Letakkan di: src/main.py
Jalankan  : python main.py  (dari folder src)

Struktur folder:
    PREDIKSI KELULUSAN MAHASISWA/
    ├── data/
    │   └── dataset_mahasiswa.csv
    ├── output/
    │   ├── hasil.txt
    │   ├── evaluasi_model.png
    │   └── model_kelulusan.pkl
    └── src/
        ├── generate_dataset.py   (jalankan dulu jika belum punya data)
        └── main.py               ← INI FILE INI
"""

# ============================================================
# IMPORT
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# ============================================================
# PATH
# ============================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data",   "dataset_mahasiswa.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HASIL_TXT   = os.path.join(OUTPUT_DIR, "hasil.txt")
GRAFIK_PATH = os.path.join(OUTPUT_DIR, "evaluasi_model.png")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "model_kelulusan.pkl")
FITUR_PATH  = os.path.join(OUTPUT_DIR, "fitur_model.pkl")

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("  SISTEM PREDIKSI KELULUSAN MAHASISWA")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {data.shape[0]} baris, {data.shape[1]} kolom")
print(data["lulus_tepat_waktu"].value_counts().rename({1:"Tepat Waktu", 0:"Terlambat"}))

# ============================================================
# 2. FITUR & TARGET
# ============================================================
# Fitur numerik → langsung dipakai
fitur_numerik = [
    "ipk", "sks", "kehadiran", "nilai_rata",
    "jumlah_cuti", "jumlah_mengulang", "lama_studi_semester",
    "status_kerja", "organisasi"
]
# Fitur kategorik → di-encode otomatis
fitur_kategorik = ["penghasilan_ortu", "jalur_masuk"]

X = data[fitur_numerik + fitur_kategorik]
y = data["lulus_tepat_waktu"]

print(f"\n[2] Fitur numerik  : {fitur_numerik}")
print(f"    Fitur kategorik: {fitur_kategorik}")

# ============================================================
# 3. SPLIT DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Split data → train: {len(X_train)}, test: {len(X_test)}")

# ============================================================
# 4. PREPROCESSING
# ============================================================
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),             fitur_numerik),
    ("cat", OneHotEncoder(handle_unknown="ignore"), fitur_kategorik),
])

# ============================================================
# 5. MODEL-MODEL YANG DIBANDINGKAN
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "SVM":                  SVC(probability=True, kernel="rbf", random_state=42),
}

print("\n[4] Melatih dan mengevaluasi model...")
hasil_list       = []
trained_pipelines = {}

for nama, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()

    hasil_list.append({
        "Model": nama, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
        "Recall": round(rec, 4), "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4), "CV-Accuracy(5fold)": round(cv, 4)
    })
    trained_pipelines[nama] = pipe
    print(f"    ✔ {nama:<22} Acc={acc:.2%}  F1={f1:.4f}  AUC={auc:.4f}")

hasil_df = pd.DataFrame(hasil_list).sort_values("F1-Score", ascending=False)

# ============================================================
# 6. PILIH MODEL TERBAIK → SIMPAN
# ============================================================
nama_terbaik  = hasil_df.iloc[0]["Model"]
model_terbaik = trained_pipelines[nama_terbaik]
joblib.dump(model_terbaik, MODEL_PATH)
joblib.dump(list(X.columns), FITUR_PATH)

# ============================================================
# 7. SIMPAN HASIL KE hasil.txt  (sesuai format proyek kamu)
# ============================================================
y_pred_best = model_terbaik.predict(X_test)
accuracy    = accuracy_score(y_test, y_pred_best)
cm          = confusion_matrix(y_test, y_pred_best)

with open(HASIL_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  LAPORAN PREDIKSI KELULUSAN MAHASISWA\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model Terpilih    : {nama_terbaik}\n")
    f.write(f"Akurasi           : {accuracy:.4f} ({accuracy:.2%})\n\n")

    f.write("Perbandingan Semua Model:\n")
    f.write("-" * 60 + "\n")
    f.write(hasil_df.to_string(index=False))
    f.write("\n\n")

    f.write(f"Confusion Matrix ({nama_terbaik}):\n")
    f.write("-" * 60 + "\n")
    f.write(f"              Prediksi\n")
    f.write(f"               Terlambat  Tepat Waktu\n")
    f.write(f"Aktual Terlambat   {cm[0][0]:>6}       {cm[0][1]:>6}\n")
    f.write(f"       Tepat Waktu {cm[1][0]:>6}       {cm[1][1]:>6}\n\n")

    f.write("Classification Report:\n")
    f.write("-" * 60 + "\n")
    f.write(classification_report(y_test, y_pred_best,
                                  target_names=["Terlambat", "Tepat Waktu"]))

print(f"\n[5] Hasil tersimpan  : {HASIL_TXT}")

# ============================================================
# 8. GRAFIK EVALUASI → output/evaluasi_model.png
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Evaluasi Model Prediksi Kelulusan Mahasiswa", fontsize=13, fontweight="bold")

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Terlambat", "Tepat Waktu"],
            yticklabels=["Terlambat", "Tepat Waktu"], ax=axes[0])
axes[0].set_title(f"Confusion Matrix\n{nama_terbaik}")
axes[0].set_xlabel("Prediksi")
axes[0].set_ylabel("Aktual")

# ROC Curve semua model
for nama, pipe in trained_pipelines.items():
    proba      = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc        = roc_auc_score(y_test, proba)
    axes[1].plot(fpr, tpr, label=f"{nama} (AUC={auc:.2f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC Curve - Perbandingan Model")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=8)

# Bar F1-Score
axes[2].barh(hasil_df["Model"], hasil_df["F1-Score"], color="steelblue")
axes[2].set_title("Perbandingan F1-Score")
axes[2].set_xlabel("F1-Score")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(GRAFIK_PATH, dpi=150)
print(f"[6] Grafik tersimpan : {GRAFIK_PATH}")
print(f"[7] Model tersimpan  : {MODEL_PATH}")

# ============================================================
# 9. CONTOH PREDIKSI MAHASISWA BARU
# ============================================================
print("\n" + "=" * 60)
print("  CONTOH PREDIKSI MAHASISWA BARU")
print("=" * 60)

def prediksi(data_dict):
    """Prediksi satu mahasiswa baru."""
    model = joblib.load(MODEL_PATH)
    fitur = joblib.load(FITUR_PATH)
    df    = pd.DataFrame([data_dict])[fitur]
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    label = "Tepat Waktu ✔" if pred == 1 else "Terlambat ✘"
    print(f"  Status          : {label}")
    print(f"  Prob Tepat Waktu: {proba[1]*100:.1f}%")
    print(f"  Prob Terlambat  : {proba[0]*100:.1f}%")
    return pred, proba

# Mahasiswa A (performa bagus)
print("\nMahasiswa A (IPK tinggi, aktif, tidak kerja):")
prediksi({
    "ipk": 3.6, "sks": 140, "kehadiran": 95, "nilai_rata": 88,
    "jumlah_cuti": 0, "jumlah_mengulang": 0, "lama_studi_semester": 7,
    "status_kerja": 0, "organisasi": 1,
    "penghasilan_ortu": "Tinggi", "jalur_masuk": "SNBP",
})

# Mahasiswa B (performa kurang)
print("\nMahasiswa B (IPK rendah, kerja full-time, banyak mengulang):")
prediksi({
    "ipk": 2.2, "sks": 75, "kehadiran": 62, "nilai_rata": 60,
    "jumlah_cuti": 2, "jumlah_mengulang": 5, "lama_studi_semester": 12,
    "status_kerja": 2, "organisasi": 0,
    "penghasilan_ortu": "Rendah", "jalur_masuk": "Mandiri",
})

print("\n" + "=" * 60)
print("  SELESAI")
print("=" * 60)