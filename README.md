
---

# Analisis Sentimen Amazon (ModernBERT + Spherical Fuzzy Sets)

Proyek ini adalah aplikasi web berbasis **Flask** untuk melakukan analisis sentimen pada ulasan produk Amazon. Sistem ini tidak hanya mengklasifikasikan sentimen menjadi **Positif**, **Netral**, atau **Negatif**, tetapi juga membandingkan kinerja antara dua pendekatan:

1. **ModernBERT Baseline:** Menggunakan arsitektur ModernBERT standar.
2. **ModernBERT + SFS (Spherical Fuzzy Sets):** Menggabungkan ModernBERT dengan logika *Fuzzy* untuk menangani ketidakpastian bahasa dengan lebih akurat.

## 📋 Fitur Utama

* **Antarmuka Modern:** Desain *Glassmorphism* yang responsif dan interaktif.
* **Dual Model:** Pilihan untuk menggunakan model standar atau model Fuzzy.
* **Riwayat Analisis:** Menyimpan hasil analisis sebelumnya ke dalam database SQLite secara otomatis.
* **Visualisasi Data:** Menampilkan grafik probabilitas sentimen secara detail.

---

## 🛠️ Persiapan Awal (Prerequisites)

Sebelum memulai, pastikan komputer Anda telah terinstal:

* [Python](https://www.python.org/) (Versi 3.8 atau lebih baru)
* [Git](https://git-scm.com/)

---

## 🚀 Langkah Instalasi

Ikuti langkah-langkah berikut secara berurutan untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Clone Repositori

Unduh kode sumber (source code) dari GitHub ke komputer Anda dengan perintah berikut di terminal/CMD:

```bash
git clone https://github.com/dandi-maulana/analisis_sentimen_amazon.git
cd analisis_sentimen_amazon

```

### 2. Siapkan Virtual Environment (Opsional tapi Disarankan)

Agar pustaka (library) tidak tercampur dengan proyek lain, disarankan membuat lingkungan virtual:

* **Untuk Windows:**
```bash
python -m venv venv
venv\Scripts\activate

```


* **Untuk Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate

```



### 3. Instal Dependensi

Instal semua pustaka yang dibutuhkan yang tercantum dalam file `requirements.txt`:

```bash
pip install -r requirements.txt

```

### 4. Unduh Model (PENTING)

Karena ukuran file model cukup besar, file tersebut tidak disimpan di GitHub. Anda harus mengunduhnya secara manual:

1. Buka tautan Google Drive berikut:
* **[Download Model (Google Drive)](https://drive.google.com/drive/folders/14O6ez4NRs9nnozhfF5L-82Wm0szl0npa?usp=drive_link)**


2. Unduh kedua file model (`.pth`) yang ada di dalamnya (biasanya bernama `modernbert_baseline_best.pth` dan `modernbert_sfs_best.pth`).
3. Pindahkan kedua file tersebut ke dalam folder **`model/`** di dalam direktori proyek Anda.

> **Catatan:** Pastikan struktur folder Anda terlihat seperti ini agar aplikasi dapat membaca model:
> ```
> analisis_sentimen_amazon/
> ├── model/
> │   ├── modernbert_baseline_best.pth
> │   └── modernbert_sfs_best.pth
> ├── templates/
> ├── app.py
> └── ...
> 
> ```
> 
> 

### 5. Jalankan Aplikasi

Setelah semua siap, jalankan aplikasi dengan perintah:

```bash
python app.py

```

Jika berhasil, Anda akan melihat pesan bahwa server berjalan (biasanya di `http://127.0.0.1:5000`).

---

## 💻 Cara Penggunaan

1. Buka browser dan kunjungi alamat **`http://127.0.0.1:5000`**.
2. **Masukkan Teks:** Ketik atau tempel ulasan produk (disarankan dalam Bahasa Inggris) pada kolom yang tersedia.
3. **Pilih Model:** Pilih "AI Standar" atau "AI Super (Fuzzy)" pada tombol pilihan.
4. **Analisis:** Klik tombol **"MULAI ANALISIS SEKARANG"**.
5. Hasil prediksi dan grafik probabilitas akan muncul di sebelah kanan. Riwayat analisis Anda juga akan tersimpan di tabel bagian bawah halaman.

---

## 📂 Struktur Proyek

* `app.py`: File utama (Back-end) yang mengatur logika Flask dan pemrosesan model.
* `templates/index.html`: Tampilan antarmuka pengguna (Front-end).
* `model/`: Folder tempat menyimpan file bobot model (`.pth`).
* `requirements.txt`: Daftar pustaka Python yang diperlukan.
* `sentiment_history.db`: Database SQLite (dibuat otomatis saat aplikasi dijalankan) untuk menyimpan riwayat.

---