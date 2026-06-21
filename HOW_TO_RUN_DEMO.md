# How to Run the Live Demo (from a laptop, no internet needed)

1. Make sure Python 3.9+ is installed.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   streamlit run app_demo.py
4. The browser will automatically open to http://localhost:8501

Files that must be in the same folder during the presentation:
- app_demo.py
- model.pkl
- ui_options.json
- pyrolysis_dataset_preprocessed.csv

If you want to retrain the model (e.g. after updating the dataset):
   python train_and_save_model.py
(this will overwrite model.pkl and ui_options.json)
