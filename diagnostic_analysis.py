"""
Diagnostic analysis requested after external AI quality-check review.
Real computation only -- no fabricated numbers.

1. DummyRegressor (mean & median strategy) under the SAME LOSO scheme as RF/ANN/SVR,
   on the FULL dataset (201 rows / 25 papers) -- baseline sanity check.
2. Per-paper MAE for Random Forest, FULL dataset, all 25 LOSO folds (not just top 5)
   -> used to build a bar chart.
3. MAE split by data_confidence (exact vs estimated) on the pooled LOSO held-out
   predictions, FULL dataset, Random Forest.
"""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("pyrolysis_dataset_preprocessed.csv")

CAT = ["feedstock_family", "catalyst_family"]
NUM = ["has_catalyst", "catalyst_loading_pct", "primary_fraction", "temperature_C"]
TARGET = "oil_yield_pct"
GROUP = "group_id"

def build_pre():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])

def run_loso(data, model, label):
    X = data[CAT + NUM]
    y = data[TARGET].values
    groups = data[GROUP].values
    conf = data["data_confidence"].values
    logo = LeaveOneGroupOut()

    per_paper = []
    all_preds, all_true, all_conf, all_paper = [], [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        pipe = Pipeline([("pre", build_pre()), ("model", model)])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        preds = pipe.predict(X.iloc[test_idx])
        paper = data.iloc[test_idx][GROUP].values[0]
        mae = mean_absolute_error(y[test_idx], preds)
        rmse = mean_squared_error(y[test_idx], preds) ** 0.5
        per_paper.append({"paper": paper, "n": len(test_idx), "mae": mae, "rmse": rmse})
        all_preds.extend(preds); all_true.extend(y[test_idx])
        all_conf.extend(conf[test_idx]); all_paper.extend([paper]*len(test_idx))

    r2 = r2_score(all_true, all_preds)
    mae_pooled = mean_absolute_error(all_true, all_preds)
    rmse_pooled = mean_squared_error(all_true, all_preds) ** 0.5
    print(f"\n[{label}] FULL dataset, LOSO, {len(per_paper)} folds")
    print(f"  Pooled  -> R2={r2:.3f}  MAE={mae_pooled:.2f}  RMSE={rmse_pooled:.2f}")
    print(f"  Per-fold mean MAE={np.mean([f['mae'] for f in per_paper]):.2f}  RMSE={np.mean([f['rmse'] for f in per_paper]):.2f}")
    return per_paper, np.array(all_preds), np.array(all_true), np.array(all_conf), np.array(all_paper), r2, mae_pooled, rmse_pooled

print("="*78)
print("1) BASELINE: DummyRegressor under LOSO, FULL dataset (201 rows/25 papers)")
print("="*78)
for strat in ["mean", "median"]:
    run_loso(df, DummyRegressor(strategy=strat), f"Dummy({strat})")

print("\n"+"="*78)
print("2) Random Forest, FULL dataset, ALL 25 per-paper LOSO folds")
print("="*78)
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42)
per_paper, preds, true, conf, paper_arr, r2, mae_pooled, rmse_pooled = run_loso(df, rf, "RandomForest")
per_paper_sorted = sorted(per_paper, key=lambda f: -f["mae"])
print("  All folds, sorted by MAE descending:")
for f in per_paper_sorted:
    print(f"    {f['paper']}: n={f['n']:2d}  MAE={f['mae']:6.2f}  RMSE={f['rmse']:6.2f}")

print("\n"+"="*78)
print("3) MAE split by data_confidence (pooled LOSO predictions, RF, FULL dataset)")
print("="*78)
mask_exact = conf == "exact"
mask_est = conf == "estimated"
mae_exact = mean_absolute_error(true[mask_exact], preds[mask_exact])
mae_est = mean_absolute_error(true[mask_est], preds[mask_est])
print(f"  Exact rows     (n={mask_exact.sum():3d}): MAE = {mae_exact:.2f}")
print(f"  Estimated rows (n={mask_est.sum():3d}): MAE = {mae_est:.2f}")
print(f"  Relative increase (estimated vs exact): {100*(mae_est-mae_exact)/mae_exact:+.1f}%")

# Save per-paper MAE for chart + feedstock lookup
feed_lookup = df.groupby("group_id")["feedstock_primary"].first().to_dict()
import json
with open("diagnostic_results.json", "w") as fh:
    json.dump({
        "per_paper": per_paper_sorted,
        "feed_lookup": feed_lookup,
        "mae_exact": mae_exact, "mae_estimated": mae_est,
        "n_exact": int(mask_exact.sum()), "n_estimated": int(mask_est.sum()),
        "rf_r2": r2, "rf_mae": mae_pooled, "rf_rmse": rmse_pooled,
    }, fh, indent=2)
print("\nSaved diagnostic_results.json")
