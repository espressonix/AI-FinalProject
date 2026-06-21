"""
Training & evaluasi model - Pyrolysis Plastic Waste ML Project
Privan Peter & Charlie Dumingan

Model utama (demo): Random Forest Regressor -> prediksi oil_yield_pct
Model pembanding (bukan untuk demo live): ANN (MLPRegressor), SVR

Validasi: Leave-One-Study-Out (LOSO) CV dikelompokkan per paper_id (group_id),
BUKAN k-fold acak -> supaya tidak ada data leakage antar baris dari paper
yang sama (baris-baris dalam 1 paper biasanya sangat berkorelasi karena
berasal dari kondisi eksperimen yang mirip).

Dijalankan 2x:
  (a) exact-only  -> sanity check pakai data dengan confidence tinggi saja
  (b) full dataset -> performa dengan seluruh data (exact + estimated)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv("pyrolysis_dataset_preprocessed.csv")

CAT_FEATURES = ["feedstock_family", "catalyst_family"]
NUM_FEATURES = ["has_catalyst", "catalyst_loading_pct", "primary_fraction", "temperature_C"]
TARGET = "oil_yield_pct"
GROUP = "group_id"

def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

def loso_eval(data, model_name, model):
    X = data[CAT_FEATURES + NUM_FEATURES]
    y = data[TARGET].values
    groups = data[GROUP].values

    logo = LeaveOneGroupOut()
    fold_results = []
    n_groups = len(set(groups))
    if n_groups < 2:
        return None

    for train_idx, test_idx in logo.split(X, y, groups):
        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        pre = build_preprocessor()
        pipe = Pipeline([("pre", pre), ("model", model)])
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
        except Exception as e:
            continue
        held_out_paper = data.iloc[test_idx][GROUP].values[0]
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        fold_results.append({
            "paper": held_out_paper,
            "n_test": len(test_idx),
            "mae": mae,
            "rmse": rmse,
            "y_true_mean": y_test.mean(),
        })

    if not fold_results:
        return None

    maes = [f["mae"] for f in fold_results]
    rmses = [f["rmse"] for f in fold_results]

    # R2 global dihitung dari gabungan prediksi semua fold (lebih stabil utk dataset kecil)
    all_preds, all_true = [], []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        pre = build_preprocessor()
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        all_preds.extend(pipe.predict(X.iloc[test_idx]))
        all_true.extend(y[test_idx])
    r2_pooled = r2_score(all_true, all_preds)

    print(f"\n[{model_name}] LOSO CV — {n_groups} grup paper, {len(fold_results)} fold valid")
    print(f"  MAE rata-rata fold : {np.mean(maes):.2f} (std {np.std(maes):.2f})")
    print(f"  RMSE rata-rata fold: {np.mean(rmses):.2f}")
    print(f"  R2 (pooled, semua prediksi held-out digabung): {r2_pooled:.3f}")
    print(f"  5 fold dengan error tertinggi:")
    for f in sorted(fold_results, key=lambda x: -x["mae"])[:5]:
        print(f"    {f['paper']}: n={f['n_test']}, MAE={f['mae']:.2f}, y_true_mean={f['y_true_mean']:.1f}")
    return {"mae": np.mean(maes), "rmse": np.mean(rmses), "r2_pooled": r2_pooled, "n_folds": len(fold_results)}

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42),
    "ANN_MLP": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=42, early_stopping=True),
    "SVR": SVR(kernel="rbf", C=10, epsilon=1.0),
}

for subset_name, subset_df in [
    ("EXACT-ONLY", df[df["data_confidence"] == "exact"].copy()),
    ("FULL (exact+estimated)", df.copy()),
]:
    print("=" * 70)
    print(f"SUBSET: {subset_name} — {len(subset_df)} baris, {subset_df['group_id'].nunique()} paper")
    print("=" * 70)
    for name, model in models.items():
        loso_eval(subset_df, name, model)
