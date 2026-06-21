"""
Live Demo - Plastic Pyrolysis Oil Yield Prediction
Final Project, Artificial Intelligence (CS641)
Privan Peter (1146056) & Charlie Dumingan (1149107)

How to run (from a laptop, no internet/cloud needed):
    pip install -r requirements.txt
    streamlit run app_demo.py

Supporting files that must be in the same folder:
    model.pkl, ui_options.json, pyrolysis_dataset_preprocessed.csv
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plastic Pyrolysis Yield Prediction", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_options():
    with open("ui_options.json") as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv("pyrolysis_dataset_preprocessed.csv")

model = load_model()
options = load_options()
df = load_dataset()

st.title("ML-Based Plastic Pyrolysis Oil Yield Prediction")
st.caption("Final Project, Artificial Intelligence (CS641) — Privan Peter (1146056) & Charlie Dumingan (1149107)")

with st.expander("Methodology note (read before interpreting results)", expanded=False):
    st.markdown("""
    This model (Random Forest) is trained on **201 rows of data extracted from
    25 scientific publications** (figure-derived data). The model is trained on
    **the entire dataset** so it can be used for this interactive demo.

    **Important:** a strict generalization evaluation (Leave-One-Study-Out CV —
    holding out one full paper as the test set) shows the model **does not yet
    generalize well to completely new studies** (pooled R² is negative). This is
    because the source studies are highly heterogeneous (feedstock-catalyst-
    temperature combinations rarely overlap across papers). Full details are in
    the report, Discussion section. The prediction below is an illustration of
    the end-to-end pipeline, not a claim of validated predictive accuracy.
    """)

st.subheader("Process Input Parameters")
col1, col2 = st.columns(2)
with col1:
    feedstock = st.selectbox(
        "Primary Feedstock Type", options["feedstock_family"],
        index=options["feedstock_family"].index("PP") if "PP" in options["feedstock_family"] else 0,
        help="The type of plastic waste/material being pyrolyzed. PP=polypropylene, PE=polyethylene, "
             "PS=polystyrene/styrofoam, PET=drink bottles, mixed_waste=unspecified mixture, "
             "biomass=non-plastic material (e.g. algae/sludge), wax=pre-processed wax.")
    temperature = st.slider(
        "Pyrolysis Temperature (°C)", 250, 700, 500, step=10,
        help="Reactor temperature when the plastic is heated without oxygen. The single most "
             "influential variable on oil/gas/char output.")
    primary_fraction = st.slider(
        "Primary Feedstock Fraction in Mixture", 0.0, 1.0, 1.0, step=0.05,
        help="If the plastic is a mix of 2 types (e.g. 70% PP + 30% PET), this value = 0.7. "
             "If the feedstock is a single, unmixed type, the value is 1.0.")
with col2:
    catalyst = st.selectbox(
        "Catalyst Category", options["catalyst_family"],
        help="An added chemical that redirects the pyrolysis reaction (usually increasing "
             "oil yield or making it cleaner/more stable). 'none' = no catalyst.")
    has_catalyst = 0 if catalyst == "none" else 1
    catalyst_loading = st.slider(
        "Catalyst Loading (%)", 0.0, 30.0, 0.0 if catalyst == "none" else 10.0, step=0.5,
        disabled=(catalyst == "none"),
        help="Catalyst weight as a percentage of the plastic weight. Only applies if "
             "a catalyst is selected (not 'none').")
    st.caption("Prediction target: **Oil Yield** = the percentage of the original plastic weight "
               "converted into liquid pyrolysis oil (which can be further processed into fuel).")

input_df = pd.DataFrame([{
    "feedstock_family": feedstock,
    "catalyst_family": catalyst,
    "has_catalyst": has_catalyst,
    "catalyst_loading_pct": catalyst_loading if catalyst != "none" else 0.0,
    "primary_fraction": primary_fraction,
    "temperature_C": temperature,
}])

def clean_label(name):
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")

if st.button("Predict Oil Yield", type="primary"):
    pred = model.predict(input_df)[0]
    pred_clipped = float(np.clip(pred, 0, 100))
    st.metric("Predicted Oil Yield", f"{pred_clipped:.1f} %")

    similar = df[df["feedstock_family"] == feedstock]
    if len(similar) > 0:
        st.caption(f"Reference: training data for feedstock '{feedstock}' has oil_yield_pct "
                   f"between {similar['oil_yield_pct'].min():.1f}% - {similar['oil_yield_pct'].max():.1f}% "
                   f"(n={len(similar)} rows, from {similar['group_id'].nunique()} papers)")

    st.subheader("Temperature Sensitivity of the Prediction (for your current input combination)")
    st.caption("This chart CHANGES based on your input — it shows how the predicted oil yield "
               "shifts when only temperature is varied, while feedstock/catalyst/fraction stay "
               "fixed at what you selected above.")
    temp_range = np.arange(250, 701, 25)
    sweep_df = pd.concat([input_df.assign(temperature_C=t) for t in temp_range], ignore_index=True)
    sweep_preds = np.clip(model.predict(sweep_df), 0, 100)

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.plot(temp_range, sweep_preds, marker="o", color="#1565C0")
    ax2.axvline(temperature, color="red", linestyle="--", alpha=0.6, label=f"Selected temperature: {temperature}°C")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Predicted Oil Yield (%)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Feature Importance — Model Overall")
    st.caption("The chart below does NOT change no matter what input you choose. This is not a bug — it "
               "shows how important each factor is to the model OVERALL (learned from the 201-row "
               "training set), not for one specific prediction. For the effect of your specific "
               "input, see the temperature sensitivity chart above.")
    pre = model.named_steps["pre"]
    rf = model.named_steps["model"]
    feat_names = [clean_label(n) for n in pre.get_feature_names_out()]
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#2E7D32")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Most Influential Features (model-level, not input-specific)")
    st.pyplot(fig)

st.divider()
st.caption("Full dataset & pipeline code available in this project's repository.")
