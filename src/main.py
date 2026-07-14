"""
main.py — Training 3 Model: Logistic Regression, Random Forest, XGBoost
Letakkan di: src/main.py
Jalankan   : python main.py (dari folder src)
"""

import os, pandas as pd, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "output")
MOD_DIR = os.path.join(BASE, "models")
PRO_DIR = os.path.join(BASE, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(MOD_DIR, exist_ok=True); os.makedirs(PRO_DIR, exist_ok=True)

print("=" * 60)
print("  SISTEM PREDIKSI KELULUSAN MAHASISWA")
print("  Model: Logistic Regression | Random Forest | XGBoost")
print("=" * 60)

# ── Load Data ────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data", "raw", "dataset_mahasiswa.csv"))
NUM = ["ipk","sks","kehadiran","nilai_rata","jumlah_cuti","jumlah_mengulang","lama_studi_semester","status_kerja","organisasi"]
CAT = ["penghasilan_ortu","jalur_masuk"]
X = df[NUM+CAT]; y = df["lulus_tepat_waktu"]
print(f"\n[1] Dataset: {df.shape[0]} baris | Target: {y.value_counts().to_dict()}")

# ── Split ────────────────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
print(f"[2] Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

# ── Preprocessing ────────────────────────────────────────────
prep = ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])

# ── SMOTE ────────────────────────────────────────────────────
X_enc = X_train.copy()
for col in CAT: X_enc[col] = LabelEncoder().fit_transform(X_enc[col])
X_res, y_res = SMOTE(random_state=42).fit_resample(X_enc, y_train)
print(f"[3] SMOTE: {y_train.value_counts().to_dict()} → {pd.Series(y_res).value_counts().to_dict()}")

# ── Training 3 Model ─────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([("prep",prep),("model",LogisticRegression(C=0.1,max_iter=1000,random_state=42))]),
    "Random Forest":       Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42))]),
    "XGBoost":             Pipeline([("prep",prep),("model",XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,eval_metric="logloss",random_state=42))]),
}

print("\n[4] Training & evaluasi model...")
hasil = []; trained = {}
for nama, pipe in models.items():
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test); ypr = pipe.predict_proba(X_test)[:,1]
    hasil.append({"Model":nama,"Accuracy":round(accuracy_score(y_test,yp),4),
                  "Precision":round(precision_score(y_test,yp),4),"Recall":round(recall_score(y_test,yp),4),
                  "F1-Score":round(f1_score(y_test,yp),4),"ROC-AUC":round(roc_auc_score(y_test,ypr),4),
                  "CV-Acc":round(cross_val_score(pipe,X_train,y_train,cv=5,scoring="accuracy").mean(),4)})
    trained[nama] = pipe
    print(f"    ✔ {nama:<25} Acc={accuracy_score(y_test,yp):.2%}  F1={f1_score(y_test,yp):.4f}  AUC={roc_auc_score(y_test,ypr):.4f}")

hdf = pd.DataFrame(hasil).sort_values("F1-Score", ascending=False)
best_name = hdf.iloc[0]["Model"]
best_pipe = trained[best_name]
print(f"\n    🥇 Model Terbaik: {best_name}")

# ── Simpan Model ─────────────────────────────────────────────
joblib.dump(trained["Logistic Regression"], os.path.join(MOD_DIR,"logistic_regression.pkl"))
joblib.dump(trained["Random Forest"],       os.path.join(MOD_DIR,"best_model.pkl"))
joblib.dump(trained["XGBoost"],             os.path.join(MOD_DIR,"xgboost_model.pkl"))
joblib.dump({"num":NUM,"cat":CAT},          os.path.join(MOD_DIR,"feature_info.pkl"))
X_test.to_csv(os.path.join(PRO_DIR,"X_test.csv"),index=False)
y_test.to_csv(os.path.join(PRO_DIR,"y_test.csv"),index=False)
X_train.to_csv(os.path.join(PRO_DIR,"X_train.csv"),index=False)
hdf.to_csv(os.path.join(PRO_DIR,"model_results.csv"),index=False)
print(f"\n[5] Semua model & data tersimpan di: models/ dan data/processed/")

# ── Grafik ───────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(18,5))
fig.suptitle("Evaluasi Model — LR | RF | XGBoost", fontsize=13, fontweight="bold")

yp_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, yp_best)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Terlambat","Tepat Waktu"],
            yticklabels=["Terlambat","Tepat Waktu"],ax=axes[0])
axes[0].set_title(f"Confusion Matrix\n{best_name}"); axes[0].set_xlabel("Prediksi"); axes[0].set_ylabel("Aktual")

for (nama,pipe), color in zip(trained.items(),["#2E74B5","#E74C3C","#27AE60"]):
    ypr = pipe.predict_proba(X_test)[:,1]; fpr,tpr,_ = roc_curve(y_test,ypr); auc=roc_auc_score(y_test,ypr)
    axes[1].plot(fpr,tpr,color=color,lw=2,label=f"{nama} (AUC={auc:.3f})")
axes[1].plot([0,1],[0,1],"k--",alpha=0.4); axes[1].set_title("ROC Curve")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].legend(fontsize=8)

metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
x = np.arange(len(metrics)); w = 0.25
for i, row in hdf.iterrows():
    idx = list(hdf["Model"]).index(row["Model"])
    axes[2].bar(x+idx*w,[row[m] for m in metrics],w,label=row["Model"],
                color=["#2E74B5","#E74C3C","#27AE60"][idx],edgecolor="white",alpha=0.85)
axes[2].set_xticks(x+w); axes[2].set_xticklabels(metrics,rotation=15); axes[2].set_ylim(0.7,1.02)
axes[2].set_title("Perbandingan Metrik"); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"evaluasi_model.png"),dpi=150)
print(f"[6] Grafik tersimpan: output/evaluasi_model.png")

# ── Hasil ────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR,"hasil.txt"),"w",encoding="utf-8") as f:
    f.write("=" * 60 + "\n  LAPORAN EVALUASI MODEL\n" + "=" * 60 + "\n\n")
    f.write(f"Model Terbaik: {best_name}\n\n")
    f.write(hdf.to_string(index=False) + "\n\n")
    f.write(f"Classification Report ({best_name}):\n")
    f.write(classification_report(y_test,yp_best,target_names=["Terlambat","Tepat Waktu"]))
print(f"[7] Laporan tersimpan: output/hasil.txt")
print("\n✅ SELESAI!")