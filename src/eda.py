"""
eda.py  ──  Exploratory Data Analysis (EDA)
Letakkan di: src/eda.py
Jalankan  : python eda.py  (dari folder src, SETELAH generate_dataset.py)

Output → output/eda_*.png dan output/eda_laporan.txt
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",   "dataset_mahasiswa.csv")
OUT_DIR    = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
TARGET = "lulus_tepat_waktu"

print("=" * 60)
print("  EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# ============================================================
# 1. INFO UMUM DATASET
# ============================================================
print("\n[1] Info Umum Dataset")
print(f"    Jumlah data   : {df.shape[0]} baris")
print(f"    Jumlah fitur  : {df.shape[1]-1} fitur + 1 target")
print(f"    Missing value : {df.isnull().sum().sum()}")
print(f"\n{df.dtypes}\n")
print(df.describe().round(2))

# ============================================================
# 2. DISTRIBUSI KELAS TARGET
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("EDA — Distribusi Target & Keseimbangan Kelas", fontsize=13, fontweight="bold")

counts = df[TARGET].value_counts()
labels = {1: "Tepat Waktu", 0: "Terlambat"}
count_labels = [labels[i] for i in counts.index]

axes[0].bar(count_labels, counts.values, color=["steelblue","tomato"], edgecolor="white")
axes[0].set_title("Jumlah Mahasiswa per Kelas")
axes[0].set_ylabel("Jumlah")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=count_labels, autopct="%1.1f%%",
            colors=["steelblue","tomato"], startangle=90)
axes[1].set_title("Proporsi Kelas Target")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_1_distribusi_target.png"), dpi=150)
print("\n[2] Grafik distribusi target tersimpan.")

# ============================================================
# 3. DISTRIBUSI FITUR NUMERIK
# ============================================================
fitur_numerik = ["ipk","sks","kehadiran","nilai_rata",
                 "jumlah_cuti","jumlah_mengulang","lama_studi_semester"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("EDA — Distribusi Fitur Numerik (per Kelas)", fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(fitur_numerik):
    for label, color in [(0,"tomato"),(1,"steelblue")]:
        subset = df[df[TARGET]==label][col]
        axes[i].hist(subset, bins=20, alpha=0.6, color=color,
                     label=labels[label], edgecolor="white")
    axes[i].set_title(col)
    axes[i].legend(fontsize=7)

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_2_distribusi_fitur_numerik.png"), dpi=150)
print("[3] Grafik distribusi fitur numerik tersimpan.")

# ============================================================
# 4. BOXPLOT FITUR NUMERIK vs TARGET
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("EDA — Boxplot Fitur Numerik vs Status Kelulusan", fontsize=13, fontweight="bold")
axes = axes.flatten()

df_plot = df.copy()
df_plot["Status"] = df_plot[TARGET].map(labels)

for i, col in enumerate(fitur_numerik):
    sns.boxplot(data=df_plot, x="Status", y=col,
                hue="Status",
                palette={"Tepat Waktu":"steelblue","Terlambat":"tomato"},
                legend=False, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel("")

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_3_boxplot.png"), dpi=150)
print("[4] Grafik boxplot tersimpan.")

# ============================================================
# 5. HEATMAP KORELASI
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
corr = df[fitur_numerik + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            mask=mask, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Heatmap Korelasi Antar Fitur Numerik", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_4_korelasi.png"), dpi=150)
print("[5] Heatmap korelasi tersimpan.")

# ============================================================
# 6. FITUR KATEGORIK vs TARGET
# ============================================================
fitur_kat = ["penghasilan_ortu", "jalur_masuk"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EDA — Fitur Kategorik vs Status Kelulusan", fontsize=13, fontweight="bold")

for i, col in enumerate(fitur_kat):
    ct = pd.crosstab(df_plot[col], df_plot["Status"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[i], color=["steelblue","tomato"],
            edgecolor="white", rot=15)
    axes[i].set_title(f"{col} vs Status Kelulusan (%)")
    axes[i].set_ylabel("Persentase (%)")
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_5_fitur_kategorik.png"), dpi=150)
print("[6] Grafik fitur kategorik tersimpan.")

# ============================================================
# 7. SIMPAN LAPORAN EDA KE TXT
# ============================================================
with open(os.path.join(OUT_DIR, "eda_laporan.txt"), "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  LAPORAN EDA — PREDIKSI KELULUSAN MAHASISWA\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Jumlah data   : {df.shape[0]} baris\n")
    f.write(f"Jumlah fitur  : {df.shape[1]-1}\n")
    f.write(f"Missing value : {df.isnull().sum().sum()}\n\n")
    f.write("Distribusi Target:\n")
    f.write(counts.rename(labels).to_string() + "\n\n")
    f.write("Statistik Deskriptif:\n")
    f.write(df[fitur_numerik].describe().round(2).to_string() + "\n\n")
    f.write("Korelasi dengan Target:\n")
    f.write(corr[TARGET].sort_values(ascending=False).round(3).to_string() + "\n")

print("[7] Laporan EDA tersimpan: output/eda_laporan.txt")
print("\n✅ EDA selesai. Cek folder output/ untuk semua grafik.")