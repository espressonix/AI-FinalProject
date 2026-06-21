import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("pyrolysis_dataset_preprocessed.csv")

def loso_r2(data, cat_feats, num_feats, model, label):
    X = data[cat_feats + num_feats]
    y = data["oil_yield_pct"].values
    groups = data["group_id"].values
    if len(set(groups)) < 2:
        print(f"  [{label}] skip - hanya 1 grup"); return
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    all_preds, all_true = [], []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if len(train_idx) < 5: continue
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        all_preds.extend(pipe.predict(X.iloc[test_idx]))
        all_true.extend(y[test_idx])
    r2 = r2_score(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    print(f"  [{label}] n={len(data)}, groups={data['group_id'].nunique()} -> R2={r2:.3f}, MAE={mae:.2f}")
    return r2

print("=== Eksperimen A: hanya baris yang punya temperature_C asli (drop yang missing) ===")
sub = df[df["temperature_missing"] == 0].copy()
loso_r2(sub, ["feedstock_family","catalyst_family"], ["has_catalyst","catalyst_loading_pct","primary_fraction","temperature_C"],
        RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42), "RF full-feat, temp-only subset")

print("\n=== Eksperimen B: RF lebih sederhana (shallow, full dataset) ===")
loso_r2(df, ["feedstock_family","catalyst_family"], ["has_catalyst","catalyst_loading_pct","primary_fraction","temperature_C"],
        RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=6, random_state=42), "RF shallow")

print("\n=== Eksperimen C: fitur lebih minimal (drop catalyst_family granular) ===")
loso_r2(df, ["feedstock_family"], ["has_catalyst","catalyst_loading_pct","primary_fraction","temperature_C"],
        RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42), "RF tanpa catalyst_family")

print("\n=== Eksperimen D: Ridge regression (linear, baseline generalisasi) ===")
loso_r2(df, ["feedstock_family","catalyst_family"], ["has_catalyst","catalyst_loading_pct","primary_fraction","temperature_C"],
        Ridge(alpha=5.0), "Ridge")

print("\n=== Eksperimen E: subset temp-only + RF shallow ===")
loso_r2(sub, ["feedstock_family","catalyst_family"], ["has_catalyst","catalyst_loading_pct","primary_fraction","temperature_C"],
        RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=3, random_state=42), "RF shallow, temp-only subset")

print("\n=== Eksperimen F: subset temp-only, fitur minimal, RF shallow ===")
loso_r2(sub, ["feedstock_family"], ["has_catalyst","catalyst_loading_pct","primary_fraction","temperature_C"],
        RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=3, random_state=42), "RF shallow minimal-feat, temp-only")
