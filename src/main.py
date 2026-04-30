import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv('../data/dataset_mahasiswa.csv')

# Fitur & target
X = data[['ipk', 'sks', 'kehadiran', 'nilai_rata', 'status_kerja', 'organisasi']]
y = data['lulus_tepat_waktu']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Simpan hasil ke file
with open('../output/hasil.txt', 'w') as f:
    f.write(f"Akurasi: {accuracy}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

print("Akurasi:", accuracy)
print("Confusion Matrix:\n", cm)

# Contoh prediksi
contoh = [[3.2, 140, 85, 80, 0, 1]]
hasil = model.predict(contoh)

print("Prediksi:", "Lulus Tepat Waktu" if hasil[0] == 1 else "Tidak")