import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("pyrolysis_dataset_preprocessed.csv")
CAT = ["feedstock_family", "catalyst_family"]
NUM = ["has_catalyst", "catalyst_loading_pct", "primary_fraction", "temperature_C"]
X = df[CAT+NUM]; y = df["oil_yield_pct"].values

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42))])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(pipe, X, y, cv=kf)
print(f"Random 5-fold CV (TIDAK dikelompokkan per paper -> rawan leakage):")
print(f"  R2 = {r2_score(y, preds):.3f}")
print(f"  MAE = {mean_absolute_error(y, preds):.2f}")
