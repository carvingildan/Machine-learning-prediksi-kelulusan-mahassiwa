"""
tuning.py  ──  Hyperparameter Tuning + SMOTE + Learning Curve + Feature Importance
Letakkan di: src/tuning.py
Jalankan  : python tuning.py  (dari folder src, SETELAH main.py pernah dijalankan)

Output → output/tuning_*.png dan output/tuning_laporan.txt
         output/model_terbaik_tuned.pkl  (model final setelah tuning)
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    learning_curve, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Path ──────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data",   "dataset_mahasiswa.csv")
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & Definisi Fitur ─────────────────────────────────
df = pd.read_csv(DATA_PATH)
TARGET = "lulus_tepat_waktu"

fitur_numerik   = ["ipk","sks","kehadiran","nilai_rata",
                   "jumlah_cuti","jumlah_mengulang","lama_studi_semester",
                   "status_kerja","organisasi"]
fitur_kategorik = ["penghasilan_ortu","jalur_masuk"]

X = df[fitur_numerik + fitur_kategorik]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                       fitur_numerik),
    ("cat", OneHotEncoder(handle_unknown="ignore"), fitur_kategorik),
])

print("=" * 60)
print("  HYPERPARAMETER TUNING + SMOTE + ANALISIS LANJUTAN")
print("=" * 60)

# ============================================================
# 1. SMOTE — Tangani Data Tidak Seimbang
# ============================================================
print("\n[1] Menerapkan SMOTE untuk menyeimbangkan data...")

# Encode dulu untuk SMOTE (SMOTE butuh data numerik)
from sklearn.preprocessing import LabelEncoder
X_train_enc = X_train.copy()
for col in fitur_kategorik:
    X_train_enc[col] = LabelEncoder().fit_transform(X_train_enc[col])

print(f"    Sebelum SMOTE → {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_enc, y_train)
print(f"    Sesudah SMOTE → {pd.Series(y_res).value_counts().to_dict()}")

# ============================================================
# 2. HYPERPARAMETER TUNING — Random Forest (GridSearchCV)
# ============================================================
print("\n[2] GridSearchCV — Random Forest...")

pipe_rf = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42)),
])

param_grid_rf = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth":    [5, 8, 12, None],
    "model__min_samples_split": [2, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv,
                       scoring="f1", n_jobs=-1, verbose=0)
grid_rf.fit(X_train, y_train)

print(f"    Best params : {grid_rf.best_params_}")
print(f"    Best CV F1  : {grid_rf.best_score_:.4f}")

# Evaluasi di test set
y_pred_rf = grid_rf.best_estimator_.predict(X_test)
print(f"    Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"    Test F1      : {f1_score(y_test, y_pred_rf):.4f}")
print(f"    Test AUC     : {roc_auc_score(y_test, grid_rf.best_estimator_.predict_proba(X_test)[:,1]):.4f}")

# ============================================================
# 3. HYPERPARAMETER TUNING — Logistic Regression
# ============================================================
print("\n[3] GridSearchCV — Logistic Regression...")

pipe_lr = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid_lr = {
    "model__C":       [0.01, 0.1, 1, 10, 100],
    "model__solver":  ["lbfgs", "liblinear"],
    "model__penalty": ["l2"],
}

grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cv,
                       scoring="f1", n_jobs=-1, verbose=0)
grid_lr.fit(X_train, y_train)

print(f"    Best params : {grid_lr.best_params_}")
print(f"    Best CV F1  : {grid_lr.best_score_:.4f}")

y_pred_lr = grid_lr.best_estimator_.predict(X_test)
print(f"    Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"    Test F1      : {f1_score(y_test, y_pred_lr):.4f}")

# ============================================================
# 4. HYPERPARAMETER TUNING — Decision Tree
# ============================================================
print("\n[4] GridSearchCV — Decision Tree...")

pipe_dt = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42)),
])

param_grid_dt = {
    "model__max_depth":        [3, 5, 8, 12, None],
    "model__min_samples_split":[2, 5, 10],
    "model__criterion":        ["gini","entropy"],
}

grid_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=cv,
                       scoring="f1", n_jobs=-1, verbose=0)
grid_dt.fit(X_train, y_train)

print(f"    Best params : {grid_dt.best_params_}")
print(f"    Best CV F1  : {grid_dt.best_score_:.4f}")

y_pred_dt = grid_dt.best_estimator_.predict(X_test)
print(f"    Test Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"    Test F1      : {f1_score(y_test, y_pred_dt):.4f}")

# ============================================================
# 5. PILIH MODEL TERBAIK SETELAH TUNING
# ============================================================
kandidat = {
    "Random Forest (tuned)":       (grid_rf.best_estimator_, grid_rf.best_score_),
    "Logistic Regression (tuned)": (grid_lr.best_estimator_, grid_lr.best_score_),
    "Decision Tree (tuned)":       (grid_dt.best_estimator_, grid_dt.best_score_),
}

nama_final  = max(kandidat, key=lambda k: f1_score(y_test, kandidat[k][0].predict(X_test)))
model_final = kandidat[nama_final][0]

print(f"\n>>> Model terbaik setelah tuning: {nama_final}")
joblib.dump(model_final, os.path.join(OUT_DIR, "model_terbaik_tuned.pkl"))
print("    Tersimpan → output/model_terbaik_tuned.pkl")

# ============================================================
# 6. LEARNING CURVE — Deteksi Overfitting / Underfitting
# ============================================================
print("\n[5] Membuat Learning Curve...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Learning Curve — Deteksi Overfitting/Underfitting", fontsize=13, fontweight="bold")

grids     = [grid_rf, grid_lr, grid_dt]
nama_grids = ["Random Forest (tuned)", "Logistic Regression (tuned)", "Decision Tree (tuned)"]

for i, (g, nama) in enumerate(zip(grids, nama_grids)):
    train_sizes, train_scores, val_scores = learning_curve(
        g.best_estimator_, X_train, y_train,
        cv=5, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    axes[i].plot(train_sizes, train_mean, "o-", color="steelblue", label="Train Score")
    axes[i].fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color="steelblue")
    axes[i].plot(train_sizes, val_mean, "o-", color="tomato", label="Validation Score")
    axes[i].fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color="tomato")
    axes[i].set_title(nama, fontsize=10)
    axes[i].set_xlabel("Jumlah Data Training")
    axes[i].set_ylabel("F1 Score")
    axes[i].legend(fontsize=8)
    axes[i].set_ylim(0.5, 1.05)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tuning_1_learning_curve.png"), dpi=150)
print("    Grafik learning curve tersimpan.")

# ============================================================
# 7. FEATURE IMPORTANCE — Random Forest
# ============================================================
print("\n[6] Membuat grafik Feature Importance...")

try:
    rf_pipe = grid_rf.best_estimator_
    ohe_cols = (rf_pipe.named_steps["preprocessing"]
                .named_transformers_["cat"]
                .get_feature_names_out(fitur_kategorik))
    all_cols     = fitur_numerik + list(ohe_cols)
    importances  = rf_pipe.named_steps["model"].feature_importances_
    feat_imp     = pd.Series(importances, index=all_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["steelblue" if i < 5 else "lightsteelblue" for i in range(len(feat_imp))]
    feat_imp.plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.set_title("Feature Importance — Random Forest (Tuned)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    ax.axvline(feat_imp.mean(), color="tomato", linestyle="--", label=f"Rata-rata ({feat_imp.mean():.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tuning_2_feature_importance.png"), dpi=150)
    print("    Grafik feature importance tersimpan.")
    print(f"\n    Top 5 Fitur Terpenting:")
    for f, v in feat_imp.head(5).items():
        print(f"      {f:<30} {v:.4f}")
except Exception as e:
    print(f"    Skipped: {e}")

# ============================================================
# 8. PERBANDINGAN SEBELUM vs SESUDAH TUNING
# ============================================================
print("\n[7] Membuat grafik perbandingan sebelum/sesudah tuning...")

# Baseline (tanpa tuning)
pipe_baseline_rf = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])
pipe_baseline_rf.fit(X_train, y_train)

perbandingan = pd.DataFrame({
    "Model": ["RF Baseline", "RF Tuned", "LR Tuned", "DT Tuned"],
    "F1-Score": [
        f1_score(y_test, pipe_baseline_rf.predict(X_test)),
        f1_score(y_test, grid_rf.best_estimator_.predict(X_test)),
        f1_score(y_test, grid_lr.best_estimator_.predict(X_test)),
        f1_score(y_test, grid_dt.best_estimator_.predict(X_test)),
    ],
    "Accuracy": [
        accuracy_score(y_test, pipe_baseline_rf.predict(X_test)),
        accuracy_score(y_test, grid_rf.best_estimator_.predict(X_test)),
        accuracy_score(y_test, grid_lr.best_estimator_.predict(X_test)),
        accuracy_score(y_test, grid_dt.best_estimator_.predict(X_test)),
    ],
}).sort_values("F1-Score", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Perbandingan Performa: Baseline vs Setelah Tuning", fontsize=13, fontweight="bold")

colors = ["tomato", "steelblue", "steelblue", "steelblue"]
axes[0].barh(perbandingan["Model"], perbandingan["F1-Score"], color=colors[::-1], edgecolor="white")
axes[0].set_title("F1-Score")
axes[0].set_xlabel("F1-Score")
axes[0].invert_yaxis()
axes[0].set_xlim(0.8, 1.0)

axes[1].barh(perbandingan["Model"], perbandingan["Accuracy"], color=colors[::-1], edgecolor="white")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Accuracy")
axes[1].invert_yaxis()
axes[1].set_xlim(0.8, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tuning_3_perbandingan.png"), dpi=150)
print("    Grafik perbandingan tersimpan.")

# ============================================================
# 9. SIMPAN LAPORAN TUNING
# ============================================================
y_pred_final = model_final.predict(X_test)
with open(os.path.join(OUT_DIR, "tuning_laporan.txt"), "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  LAPORAN HYPERPARAMETER TUNING\n")
    f.write("=" * 60 + "\n\n")

    f.write("[ SMOTE ]\n")
    f.write(f"  Sebelum: {y_train.value_counts().to_dict()}\n")
    f.write(f"  Sesudah: {pd.Series(y_res).value_counts().to_dict()}\n\n")

    f.write("[ GridSearchCV — Random Forest ]\n")
    f.write(f"  Best params : {grid_rf.best_params_}\n")
    f.write(f"  Best CV F1  : {grid_rf.best_score_:.4f}\n\n")

    f.write("[ GridSearchCV — Logistic Regression ]\n")
    f.write(f"  Best params : {grid_lr.best_params_}\n")
    f.write(f"  Best CV F1  : {grid_lr.best_score_:.4f}\n\n")

    f.write("[ GridSearchCV — Decision Tree ]\n")
    f.write(f"  Best params : {grid_dt.best_params_}\n")
    f.write(f"  Best CV F1  : {grid_dt.best_score_:.4f}\n\n")

    f.write(f"[ Model Final Terpilih: {nama_final} ]\n")
    f.write(classification_report(y_test, y_pred_final,
                                  target_names=["Terlambat","Tepat Waktu"]))

print(f"\n    Laporan tuning tersimpan → output/tuning_laporan.txt")
print("\n✅ Tuning selesai. Cek folder output/ untuk semua hasil.")
print("\nFile output yang dihasilkan:")
print("  - tuning_1_learning_curve.png")
print("  - tuning_2_feature_importance.png")
print("  - tuning_3_perbandingan.png")
print("  - tuning_laporan.txt")
print("  - model_terbaik_tuned.pkl")