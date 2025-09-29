import streamlit as st
import pickle
import json
import numpy as np

# --- Load model ---
model = pickle.load(open("home_prices_model.pickle", "rb"))

# --- Load columns.json ---
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Features
numeric_features = data_columns[:3]  # ['total_sqft', 'bath', 'bhk']
location_columns = data_columns[3:]  # all locations

# --- Streamlit UI ---
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="centered")

st.markdown(
    """
    <h2 style="text-align:center; color:#2E86C1;">🏡 House Price Prediction App</h2>
    <p style="text-align:center; color:gray;">Enter house details to get an estimated price</p>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# Inputs
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("📐 Area (sqft)", min_value=200, max_value=999999, step=50, value=1000)
    bhk = st.number_input("🛋 Number of BHK", min_value=1, max_value=20, step=1, value=2)

with col2:
    bedroom = st.number_input("🛏 Number of Bedrooms", min_value=1, max_value=30, step=1, value=2)
    bathroom = st.number_input("🛁 Number of Bathrooms", min_value=1, max_value=30, step=1, value=2)

# Location dropdown
location = st.selectbox("📍 Select Location", sorted(location_columns))

st.write("---")

# Predict Button
if st.button("🔮 Predict Price", use_container_width=True):
    # Prepare input vector
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bathroom
    x[2] = bhk
    # one-hot encode location
    if location in location_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # Predict
    prediction = model.predict([x])[0]

    st.success(f"💰 Estimated Price: **₹ {prediction:,.2f}**")
    st.balloons()
