"""
Latih Random Forest final (pakai SELURUH dataset, bukan held-out) dan simpan
sebagai model.pkl untuk dipakai oleh demo app (app_demo.py).

PENTING (baca CATATAN_METODOLOGI.md): model ini dilatih dari seluruh 201 baris
data demi kebutuhan demo live. Estimasi performa generalisasi yang JUJUR
(Leave-One-Study-Out CV) ada di file loso_results.txt - JANGAN klaim R2 dari
training-set-fit sebagai ukuran akurasi model di laporan.
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CAT_FEATURES = ["feedstock_family", "catalyst_family"]
NUM_FEATURES = ["has_catalyst", "catalyst_loading_pct", "primary_fraction", "temperature_C"]
TARGET = "oil_yield_pct"

df = pd.read_csv("pyrolysis_dataset_preprocessed.csv")
X = df[CAT_FEATURES + NUM_FEATURES]
y = df[TARGET]

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])
model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42)
pipe = Pipeline([("pre", pre), ("model", model)])
pipe.fit(X, y)

joblib.dump(pipe, "model.pkl")

# simpan opsi kategori unik untuk dropdown UI
import json
options = {
    "feedstock_family": sorted(df["feedstock_family"].unique().tolist()),
    "catalyst_family": sorted(df["catalyst_family"].unique().tolist()),
}
with open("ui_options.json", "w") as f:
    json.dump(options, f, indent=2)

print("Model disimpan: model.pkl")
print("Opsi UI:", options)
print(f"Dilatih dari {len(df)} baris, {df['group_id'].nunique()} paper.")
