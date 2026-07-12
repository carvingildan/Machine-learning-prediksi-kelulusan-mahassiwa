"""
JALANKAN SEKALI untuk membuat dataset_mahasiswa.csv di folder data/
Letakkan file ini di: src/generate_dataset.py
Jalankan dari folder src: python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

# ---- Fitur mahasiswa ----
ipk                    = np.clip(np.random.normal(3.0, 0.45, N), 1.5, 4.0).round(2)
sks                    = np.clip(np.random.normal(120, 25, N), 20, 160).astype(int)
kehadiran              = np.clip(np.random.normal(85, 10, N), 40, 100).round(1)
nilai_rata             = np.clip(np.random.normal(78, 8, N), 50, 100).round(1)
jumlah_cuti            = np.random.poisson(0.3, N)
jumlah_mengulang       = np.random.poisson(1.0, N)
lama_studi_semester    = np.clip(np.random.normal(8, 1.5, N), 6, 14).astype(int)
penghasilan_ortu       = np.random.choice(["Rendah", "Menengah", "Tinggi"], N, p=[0.35, 0.45, 0.20])
status_kerja           = np.random.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1])   # 0=tdk kerja, 1=part, 2=full
jalur_masuk            = np.random.choice(["SNBP", "SNBT", "Mandiri"], N, p=[0.3, 0.4, 0.3])
organisasi             = np.random.choice([0, 1], N, p=[0.55, 0.45])          # 0=tdk aktif, 1=aktif

# ---- Skor laten → probabilitas lulus tepat waktu ----
skor = (
    (ipk - 2.0) * 2.2
    + (sks - 100) * 0.03
    + (kehadiran - 70) * 0.05
    + (nilai_rata - 70) * 0.04
    - jumlah_cuti * 1.2
    - jumlah_mengulang * 0.9
    - (lama_studi_semester - 8) * 0.6
    + organisasi * 0.3
    - (status_kerja == 2) * 0.8
    - (status_kerja == 1) * 0.2
    + np.random.normal(0, 1.0, N)
    - 0.9
)

lulus_tepat_waktu = (1 / (1 + np.exp(-skor)) > 0.5).astype(int)

df = pd.DataFrame({
    "ipk":                  ipk,
    "sks":                  sks,
    "kehadiran":            kehadiran,
    "nilai_rata":           nilai_rata,
    "jumlah_cuti":          jumlah_cuti,
    "jumlah_mengulang":     jumlah_mengulang,
    "lama_studi_semester":  lama_studi_semester,
    "penghasilan_ortu":     penghasilan_ortu,
    "status_kerja":         status_kerja,   # 0/1/2
    "jalur_masuk":          jalur_masuk,
    "organisasi":           organisasi,     # 0/1
    "lulus_tepat_waktu":    lulus_tepat_waktu,  # TARGET: 1 = ya, 0 = tidak
})

output_path = os.path.join(os.path.dirname(__file__), "../data/dataset_mahasiswa.csv")
df.to_csv(output_path, index=False)
print(f"Dataset dibuat: {output_path}")
print(f"Total data : {len(df)} mahasiswa")
print(df["lulus_tepat_waktu"].value_counts().rename({1: "Tepat Waktu", 0: "Terlambat"}))