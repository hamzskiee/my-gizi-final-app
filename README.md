# My Gizi - Dashboard Analisis Nutrisi & AI

**My Gizi** adalah aplikasi web modern untuk menganalisis data nutrisi makanan menggunakan Data Science dan Machine Learning. Aplikasi ini dibangun menggunakan Python dan Streamlit.

### 1.Dashboard Analitik
* **Visualisasi Interaktif:** Scatter Plot Neon untuk melihat korelasi antar nutrisi.
* **Analisis Makro:** Donut chart untuk proporsi kategori dan Bar chart untuk rata-rata makronutrisi (Protein, Lemak, Karbo).
* **Peringkat Makanan:** Melihat Top 5 makanan dengan Kalori dan Protein tertinggi secara otomatis.
* **Filter Canggih:** Filter data berdasarkan Produsen dan Rentang Kalori.

### 2. AI Nutritionist (Klasifikasi)
* **Algoritma Random Forest:** Memprediksi apakah makanan termasuk kategori **Rendah**, **Sedang**, atau **Tinggi Kalori**.
* **Prediksi Real-time:** Masukkan data nutrisi (protein, lemak, gula, dll) dan dapatkan hasil analisis instan.
* **Akurasi Realistis:** Model dioptimalkan agar tidak overfitting, dengan performa yang stabil (~92%).

### 3. Segmentasi Cerdas (Clustering)
* **Unsupervised Learning (K-Means):** Mengelompokkan makanan secara otomatis ke dalam cluster berdasarkan kemiripan pola gizi.
* **Profil Cluster:** Menganalisis karakteristik setiap kelompok menggunakan Radar Chart.

## Teknologi yang Digunakan
* **Framework:** Streamlit
* **Data Processing:** Pandas
* **Visualization:** Plotly Express, Plotly Graph Objects
* **Machine Learning:** Scikit-Learn (Random Forest, K-Means)
