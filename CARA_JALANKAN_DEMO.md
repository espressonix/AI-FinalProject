# Cara Menjalankan Demo Live (dari laptop, tanpa internet)

1. Pastikan Python 3.9+ terinstall.
2. Install dependency:
   pip install -r requirements.txt
3. Jalankan:
   streamlit run app_demo.py
4. Browser otomatis terbuka ke http://localhost:8501

File yang harus ada di folder yang sama saat presentasi:
- app_demo.py
- model.pkl
- ui_options.json
- pyrolysis_dataset_preprocessed.csv

Jika ingin retrain model (misal setelah update dataset):
   python train_and_save_model.py
(ini akan menimpa model.pkl dan ui_options.json)
