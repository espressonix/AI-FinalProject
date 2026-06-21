# Penjelasan Detail Demo Streamlit — Prediksi Yield Pirolisis Plastik

Final Project AI — Privan Peter (1146056) & Charlie Dumingan (1149107)

Dokumen ini dibuat supaya Anda berdua bisa menjelaskan demo ke profesor tanpa membaca dari skrip atau menebak-nebak saat ditanya. Bacaan: 10-15 menit. Setelah baca ini, Anda akan tahu persis apa yang terjadi di balik setiap tombol, setiap grafik, dan kenapa setiap pilihan desain dibuat seperti itu.

---

## 1. Yang paling penting dipahami DULU, sebelum detail teknis

Demo ini **bukan bukti bahwa model akurat**. Demo ini adalah **ilustrasi pipeline end-to-end** — cara menunjukkan bagaimana input proses pirolisis diterjemahkan jadi prediksi oleh model yang sudah dilatih. Evaluasi yang menentukan apakah model ini *benar-benar bisa dipercaya* ada di laporan (Bagian 4/Hasil, Tabel 1), bukan di demo.

Kenapa ini penting disampaikan duluan ke profesor: kalau Anda mendemokan ini lalu profesor tanya "jadi modelnya bagus?", jawaban yang salah adalah "iya, lihat prediksinya masuk akal". Jawaban yang benar: **"Model ini dilatih dari seluruh 201 baris data supaya demo bisa berjalan, tapi evaluasi LOSO yang lebih ketat di laporan menunjukkan R² pooled -0,411 — model belum generalisasi baik ke studi yang benar-benar baru. Demo ini menunjukkan pipeline-nya berjalan, bukan klaim akurasi."** Ini sudah tertulis di expander "Catatan metodologi" di demo — jadi kalau ditanya, tunjuk langsung ke situ, jangan improvisasi.

---

## 2. Apa yang dimuat saat demo dibuka

Tiga file di-load sekali di awal dan di-cache (`@st.cache_resource` / `@st.cache_data` di baris 22-37 `app_demo.py`):

| File | Isi | Dipakai untuk |
|---|---|---|
| `model.pkl` | Pipeline RF terlatih (preprocessor + RandomForestRegressor) | Membuat prediksi |
| `ui_options.json` | Daftar nilai unik untuk dropdown (feedstock_family, catalyst_family) | Mengisi pilihan di selectbox |
| `pyrolysis_dataset_preprocessed.csv` | Dataset 201 baris yang sama dipakai untuk training | Menghitung rentang referensi (lihat Bagian 4) |

Model **TIDAK dilatih ulang saat demo dijalankan** — `model.pkl` sudah hasil training sebelumnya (lihat `CARA_JALANKAN_DEMO.md`: retrain hanya lewat `train_and_save_model.py`, dan itu menimpa file). Jadi kalau ditanya "modelnya dilatih kapan?", jawabannya: sebelum demo, dengan seluruh dataset, hasilnya disimpan ke `model.pkl`.

---

## 3. Empat input — apa artinya dan dari mana asalnya

| Input UI | Kolom dataset | Tipe | Catatan |
|---|---|---|---|
| Jenis Feedstock Utama | `feedstock_family` | kategorikal | PP, PE, PS, PET, mixed_waste, biomass, wax |
| Suhu Pirolisis (°C) | `temperature_C` | numerik, 250-700 | variabel paling berpengaruh (lihat Bagian 6) |
| Fraksi Feedstock Utama | `primary_fraction` | numerik, 0-1 | 1.0 = feedstock murni, <1.0 = campuran |
| Kategori Katalis | `catalyst_family` | kategorikal | "none" = tanpa katalis |
| Catalyst Loading (%) | `catalyst_loading_pct` | numerik, 0-30 | otomatis 0 dan disabled kalau katalis = "none" |

Detail kecil yang sering jadi pertanyaan: ada kolom keenam, `has_catalyst`, yang **tidak muncul di UI** — itu dihitung otomatis dari pilihan katalis (`0 if catalyst == "none" else 1`, baris 79). Jadi sebenarnya model menerima 6 fitur, bukan 5; satu di antaranya derived, bukan input langsung.

---

## 4. Tombol "Prediksi Oil Yield" — apa yang terjadi di balik layar

Saat tombol diklik (baris 100-103):

```python
pred = model.predict(input_df)[0]
pred_clipped = float(np.clip(pred, 0, 100))
```

Tiga hal yang terjadi dalam dua baris ini:
1. `input_df` (1 baris, 6 kolom dari input Anda) masuk ke pipeline yang sama dengan training: imputer median untuk numerik, StandardScaler, OneHotEncoder untuk kategorikal — semua otomatis lewat objek `pre` yang tersimpan di dalam `model.pkl`.
2. RandomForestRegressor menghasilkan angka prediksi mentah.
3. `np.clip(pred, 0, 100)` — prediksi dipotong ke rentang 0-100% karena oil yield secara fisik tidak mungkin negatif atau di atas 100%. **Ini penting**: kalau model mentah memprediksi misalnya -3% atau 105%, itu dipaksa jadi 0% atau 100%. Kalau profesor melihat hasil prediksi mepet di 0 atau 100, itu pertanda model sedang ekstrapolasi di luar kepercayaannya — bukan hasil "bersih".

Setelah itu, dua output tambahan ditampilkan, dijelaskan di Bagian 5 dan 6.

---

## 5. Output: rentang referensi dari data training (baris 105-109)

```python
similar = df[df["feedstock_family"] == feedstock]
```

Ini **bukan bagian dari model** — ini cuma query langsung ke dataset CSV, mencari baris-baris dengan `feedstock_family` yang sama dengan input Anda, lalu menampilkan min-max `oil_yield_pct` dari baris-baris itu. Tujuannya: kasih konteks kasar — "data training untuk PP itu yield-nya antara X% - Y%" — supaya audiens bisa menilai sendiri apakah prediksi model masuk akal dibanding histori, tanpa harus percaya buta ke satu angka prediksi.

Kalau profesor tanya "ini hasil model juga?" — jawabannya **bukan**, ini statistik deskriptif dari data mentah, dipakai sebagai pembanding visual saja.

---

## 6. Grafik sensitivitas suhu (baris 111-125) — yang paling sering disalahpahami

```python
temp_range = np.arange(250, 701, 25)
sweep_df = pd.concat([input_df.assign(temperature_C=t) for t in temp_range], ...)
sweep_preds = np.clip(model.predict(sweep_df), 0, 100)
```

Logikanya: ambil input Anda (feedstock, katalis, fraksi — semua TETAP sama persis seperti yang dipilih), lalu **hanya suhu yang digeser** dari 250°C sampai 700°C dengan step 25°C, dan model dipanggil ulang untuk setiap nilai suhu itu. Hasilnya satu kurva: bagaimana prediksi oil yield berubah kalau cuma suhu yang berubah, kombinasi lain dikunci.

Ini grafik yang **berubah sesuai input Anda** — kalau Anda ganti feedstock jadi PET misalnya, lalu klik prediksi lagi, kurva ini akan berbeda bentuknya, karena RF menangkap interaksi non-linear antara suhu dan feedstock.

Poin yang perlu diwaspadai saat presentasi: ini **bukan eksperimen fisik**, ini cuma model RF dipanggil 19 kali (jumlah titik di `temp_range`) dengan input sintetis. Kalau profesor tanya "ini hasil eksperimen lab?", jawabannya tegas: bukan, ini simulasi dari model yang sudah dilatih, bukan data baru.

---

## 7. Grafik feature importance (baris 127-142) — yang paling sering disalahpahami KEDUA

```python
rf = model.named_steps["model"]
importances = rf.feature_importances_
```

Beda fundamental dengan grafik di Bagian 6: grafik ini **TIDAK berubah** berapa kali pun Anda ganti input dan klik prediksi ulang. Ini bukan bug, dan kalau Anda tidak menjelaskan ini di awal, audiens bisa salah kira sistemnya error/diam.

Alasannya: `feature_importances_` adalah properti dari MODEL itu sendiri (seberapa sering dan seberapa efektif setiap fitur dipakai untuk split di seluruh 300 pohon RF, dihitung sekali saat training dari 201 baris data). Ini mengukur kepentingan fitur **secara umum/global**, bukan untuk satu prediksi spesifik Anda. Untuk efek satu fitur ke satu prediksi spesifik, itu yang ditunjukkan grafik sensitivitas suhu di Bagian 6 — tapi hanya untuk suhu, bukan untuk fitur lain (feedstock, katalis, dll. tidak ada sweep serupa di demo ini).

Ini juga yang jadi sumber Tabel 3 di laporan — angka-angkanya identik, hanya beda cara penyajian (laporan: tabel; demo: bar chart top-10).

---

## 8. Pemetaan kode demo ke bagian laporan

| Bagian demo | Bagian laporan |
|---|---|
| Expander "Catatan metodologi" | Tabel 1 (k-fold 0,435 vs LOSO -0,411), Bagian Diskusi |
| Input parameter proses | Bagian Metode (deskripsi fitur) |
| Grafik feature importance | Tabel 3 (feature importance RF) |
| Grafik sensitivitas suhu | Tidak ada tabel langsung — ini ilustrasi tambahan, bukan hasil yang dilaporkan di tabel manapun |
| Rentang referensi data training | Lampiran A (ringkasan 25 paper) — sumber datanya sama |

---

## 9. Pertanyaan yang kemungkinan ditanyakan profesor, dan jawaban siap pakai

**"Kalau LOSO R²-nya negatif, kenapa masih bikin demo prediksi?"**
Jawab: Tujuan demo bukan klaim akurasi tinggi, tapi menunjukkan pipeline lengkap (preprocessing → model → output) berjalan end-to-end, dan menjadi alat diskusi untuk *kenapa* generalisasi gagal — misalnya dengan menunjukkan rentang sensitivitas yang kadang tidak masuk akal secara fisik untuk kombinasi feedstock-suhu yang jarang muncul di data training.

**"Model dilatih dengan data apa, training/test split-nya gimana?"**
Jawab: Untuk demo, model dilatih dengan SELURUH 201 baris (tidak ada hold-out), karena tujuannya alat ilustrasi, bukan model yang diklaim tervalidasi. Evaluasi yang valid (LOSO, per paper) dilakukan terpisah di `train_models.py` dan dilaporkan di Tabel 1 — bukan model yang sama persis dengan `model.pkl` di demo, tapi pipeline dan hyperparameter-nya identik.

**"Kenapa hasilnya di-clip 0-100?"**
Jawab: Karena oil yield secara fisik adalah persentase massa, jadi tidak mungkin negatif atau di atas 100%; RF kadang mengekstrapolasi sedikit di luar rentang itu untuk kombinasi input yang jarang ada di data training.

**"Apa beda dua grafik di bawah hasil prediksi?"**
Jawab: Grafik sensitivitas suhu berubah sesuai input Anda (efek lokal, satu fitur, satu kombinasi input). Grafik feature importance tetap konstan (properti global model, dari seluruh data training, tidak bergantung pada satu prediksi spesifik).

**"Rentang referensi itu hasil model juga?"**
Jawab: Bukan, itu statistik min-max langsung dari data training yang feedstock-nya sama, dipakai sebagai pembanding kasar terhadap hasil prediksi.

---

## 10. Hal yang harus DIHINDARI saat presentasi

- Jangan bilang "model ini akurat" atau "model ini tervalidasi" — gunakan kata "ilustrasi pipeline" atau "demo end-to-end".
- Jangan diam saja kalau grafik feature importance tidak berubah setelah ganti input — itu memang sengaja, sebutkan secara proaktif sebelum ditanya.
- Jangan klaim grafik sensitivitas suhu sebagai "hasil eksperimen" — itu hasil pemanggilan model berulang dengan input sintetis.
- Jangan menjalankan retrain (`train_and_save_model.py`) secara live di depan kelas — tidak perlu dan berisiko macet/error tanpa alasan kuat.
