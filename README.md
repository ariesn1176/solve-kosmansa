<p align="center"><img src="https://socialify.git.ci/ariesn1176/solve-kosmansa/image?font=Source+Code+Pro&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Charlie+Brown&amp;theme=Dark" alt="project-image"></p>
<h1 align="center" id="title">KOSMANSA SOLVE Model 🚀</h1>

**Sales Optimization via LSTM-based Variable Estimation**

Model Prediksi Penjualan Produk Pangan menggunakan Arsitektur *Long Short-Term Memory* (LSTM).

---

## 📌 Deskripsi Proyek
**KOSMANSA SOLVE** adalah sistem peramalan cerdas yang dirancang untuk mengoptimalkan stok barang konsinyasi. Dengan memanfaatkan data historis penjualan harian, model ini memberikan estimasi kebutuhan stok di masa depan untuk mencegah kerugian akibat barang basi (*overstock*) atau kehilangan potensi penjualan (*stockout*).

### Fitur Utama:
- **Time Series Forecasting:** Prediksi berbasis urutan waktu harian.
- **High Accuracy:** Data time series tanpa celah.
- **Hybrid Architecture:** Kombinasi LSTM dan Dense Layer untuk menangkap pola penjualan yang kompleks.

---

## 🛠️ Teknologi yang Digunakan
Proyek ini dibangun menggunakan pustaka Python utama:
- `TensorFlow/Keras` (Model Deep Learning)
- `Pandas` & `NumPy` (Pengolahan Data)
- `Scikit-Learn` (Scaling & Evaluasi)
- `Matplotlib` (Visualisasi Hasil)

---

## 🚀 Cara Menjalankan

### 1. Clone Repositori
Buka terminal atau command prompt, lalu jalankan:
```
git clone https://github.com/ariesn1176/solve-kosmansa.git
cd solve-kosmansa
```
### 2. Siapkan Dataset
Pastikan file dataset `All24.csv` telah terdownload.

Pastikan file tersebut terletak di folder yang sama dengan file `main.py`.

### 3. Install Dependencies
Instal semua library yang dibutuhkan dengan perintah berikut:
```
pip install pandas numpy tensorflow matplotlib scikit-learn
```

### 4. Jalankan Model
Jika menggunakan Jupyter Notebook atau Google Colab, buka file `.ipynb` dan jalankan sel kode secara berurutan. Untuk memuat data di dalam kode Python:
```
import pandas as pd
df = pd.read_csv('All24.csv')
print("Data Berhasil Dimuat:")
print(df.head())
```
---
### 👤 Penulis
##### Achmad Aries Nazali Program Studi Teknik Informatika Fakultas Teknologi dan Desain Institut Teknologi dan Bisnis Asia Malang

#### Dosen Pembimbing: Dr. Vivi Aida Fitria, S.Si, M.Si.

###### © 2026 KOSMANSA SOLVE Model 

