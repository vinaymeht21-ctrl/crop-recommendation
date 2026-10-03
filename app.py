from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "crop_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "Crop_recommendation.csv")


model, data = load_model(), load_data()

st.title("What should you plant?")
st.write(
    "Enter your soil and weather conditions. A Random Forest trained on 2,200 "
    "records ranks the 22 crops that suit them best."
)

# (label, column, step, help text)
inputs = [
    ("Nitrogen (N)", "N", 1, "Nitrogen content in the soil"),
    ("Phosphorus (P)", "P", 1, "Phosphorus content in the soil"),
    ("Potassium (K)", "K", 1, "Potassium content in the soil"),
    ("Temperature (°C)", "temperature", 0.5, "Average temperature"),
    ("Humidity (%)", "humidity", 1.0, "Relative humidity"),
    ("Soil pH", "ph", 0.1, "Acidity of the soil"),
    ("Rainfall (mm)", "rainfall", 1.0, "Rainfall in mm"),
]

defaults = {"N": 90, "P": 42, "K": 43, "temperature": 21.0,
            "humidity": 82.0, "ph": 6.5, "rainfall": 203.0}

values = {}
cols = st.columns(2)
for i, (label, col, step, tip) in enumerate(inputs):
    lo, hi = float(data[col].min()), float(data[col].max())
    is_int = isinstance(step, int)
    with cols[i % 2]:
        values[col] = st.number_input(
            label,
            min_value=int(lo) if is_int else lo,
            max_value=int(hi) if is_int else hi,
            value=defaults[col] if not is_int else int(defaults[col]),
            step=step,
            help=f"{tip}. Range in training data: {lo:.1f} to {hi:.1f}",
        )

if st.button("Recommend a crop", type="primary", use_container_width=True):
    row = pd.DataFrame([values], columns=FEATURES)
    proba = pd.Series(model.predict_proba(row)[0], index=model.classes_)
    top = proba.sort_values(ascending=False).head(5)

    st.subheader(f"Best match: {top.index[0].title()}")
    st.caption(f"Model confidence: {top.iloc[0]:.0%}")

    chart = (top * 100).round(1).rename("Confidence (%)")
    chart.index = [c.title() for c in chart.index]
    st.bar_chart(chart, horizontal=True)

with st.expander("About this project"):
    st.write(
        "Built with scikit-learn and Streamlit on the Kaggle Crop Recommendation "
        "dataset. Cross-validated accuracy is about 99%. Treat results as a "
        "starting point, not agronomic advice."
    )
