# app.py - Updated Health Insurance Claim Prediction App (full file)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json

st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title("Health Insurance Claim Prediction App")
st.write("Input the following details and get a predicted insurance claim amount.")

# ART_DIR: change this to the folder where your artifacts live, if needed
ART_DIR = Path(".")  

@st.cache_resource
def load_artifacts(art_dir: Path):
    # Required artifacts (adjust names if your files have different names)
    model = joblib.load(art_dir / "best_model.pkl")
    scaler = joblib.load(art_dir / "scaler.pkl")
    le_gender = joblib.load(art_dir / "label_encoder_gender.pkl")
    le_diabetic = joblib.load(art_dir / "label_encoder_diabetic.pkl")
    le_smoker = joblib.load(art_dir / "label_encoder_smoker.pkl")

    # Optional artifacts
    ohe_region = None
    feature_cols = None
    if (art_dir / "onehot_region.pkl").exists():
        try:
            ohe_region = joblib.load(art_dir / "onehot_region.pkl")
        except Exception:
            ohe_region = None

    if (art_dir / "feature_columns.json").exists():
        try:
            with open(art_dir / "feature_columns.json", "r") as f:
                feature_cols = json.load(f)
        except Exception:
            feature_cols = None

    return model, scaler, le_gender, le_diabetic, le_smoker, ohe_region, feature_cols

# Load artifacts and error out gracefully if missing
try:
    model, scaler, le_gender, le_diabetic, le_smoker, ohe_region, feature_cols = load_artifacts(ART_DIR)
except FileNotFoundError as e:
    st.error(f"Artifact file not found. Make sure your artifacts are in {ART_DIR.resolve()}. Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# Build human-friendly options for selectboxes
def readable_options_from_encoder(le, fallback):
    """Return readable list of labels for selectbox.
       If encoder classes_ look numeric like '0','1', return fallback human labels."""
    classes = list(getattr(le, "classes_", fallback))
    classes_str = [str(x) for x in classes]
    # if classes are numeric strings or numeric values like 0/1, fallback to human labels
    if set(classes_str) <= {"0", "1", "0.0", "1.0"}:
        return fallback
    return classes_str

gender_options = readable_options_from_encoder(le_gender, ["female", "male"])
smoker_options = readable_options_from_encoder(le_smoker, ["no", "yes"])
diabetic_options = readable_options_from_encoder(le_diabetic, ["no", "yes"])

region_options = None
if ohe_region is not None:
    try:
        region_options = list(ohe_region.categories_[0])
    except Exception:
        region_options = None

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        bmi = st.number_input("BMI", min_value=5.0, max_value=70.0, value=24.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
    with col2:
        blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=250, value=120, step=1)
        gender = st.selectbox("Gender", gender_options, index=0)
        diabetic_label = st.selectbox("Diabetic", diabetic_options, index=0)
        smoker_label = st.selectbox("Smoker", smoker_options, index=0)
        if region_options is not None:
            region = st.selectbox("Region", region_options, index=0)
        else:
            region = None

    submitted = st.form_submit_button("Predict Payment")

# Mapping helpers (try encoders first, then fallbacks)
def map_with_encoder_or_fallback(value, le, fallback_map):
    """Try to transform using LabelEncoder 'le'. If fails, fallback_map(value)."""
    try:
        return int(le.transform([str(value)])[0])
    except Exception:
        return fallback_map(value)

if submitted:
    # Build input DataFrame (labels for now)
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [blood_pressure],
        "diabetic": [diabetic_label],
        "children": [children],
        "smoker": [smoker_label]
    })

    # Convert gender
    input_df["gender"] = input_df["gender"].astype(str).apply(
        lambda x: map_with_encoder_or_fallback(x, le_gender, lambda v: 1 if str(v).strip().lower() in ("male","m") else 0)
    )

    # Convert smoker & diabetic
    input_df["smoker"] = input_df["smoker"].astype(str).apply(
        lambda x: map_with_encoder_or_fallback(x, le_smoker, lambda v: 1 if str(v).strip().lower() in ("yes","1","true") else 0)
    )
    input_df["diabetic"] = input_df["diabetic"].astype(str).apply(
        lambda x: map_with_encoder_or_fallback(x, le_diabetic, lambda v: 1 if str(v).strip().lower() in ("yes","1","true") else 0)
    )

    # Scale numeric columns (scaler expects 2D array). Try to use column names, else fallback.
    numeric_cols = ["age", "bmi", "bloodpressure", "children"]
    try:
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    except Exception:
        # fallback: pass numeric array in same order
        try:
            arr = np.array([[age, bmi, blood_pressure, children]], dtype=float)
            scaled = scaler.transform(arr)[0]
            for i, c in enumerate(numeric_cols):
                input_df.at[0, c] = scaled[i]
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            st.stop()

    # Encode region (if OHE exists)
    if ohe_region is not None and region is not None:
        try:
            region_full = ohe_region.transform([[region]])  # may include dropped first column
            # append region columns named consistently to avoid name collisions
            for i, val in enumerate(region_full[0]):
                input_df[f"__region_{i}"] = val
        except Exception:
            # warn but continue
            st.warning("Region encoding failed — continuing without region features.")

    # Align input to model's expected features
    final_feature_list = None
    if hasattr(model, "feature_names_in_"):
        try:
            final_feature_list = list(model.feature_names_in_)
        except Exception:
            final_feature_list = None

    if final_feature_list is None and (ART_DIR / "feature_columns.json").exists():
        try:
            with open(ART_DIR / "feature_columns.json", "r") as f:
                final_feature_list = json.load(f)
        except Exception:
            final_feature_list = None

    if final_feature_list is not None:
        # ensure all required columns are present (add zeros for those missing)
        for col in final_feature_list:
            if col not in input_df.columns:
                input_df[col] = 0.0
        # reorder
        input_df = input_df[final_feature_list]

    # Finally predict
    try:
        pred = model.predict(input_df)[0]
        st.success(f"Estimated Insurance Payment Amount: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Input features used (for debugging):")
        st.write(input_df)
