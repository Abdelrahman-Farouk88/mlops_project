import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"  # Change after deployment

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="centered"
)

st.title("💧 Water Potability Prediction")
st.markdown("Predict whether water is **safe to drink** based on water quality parameters.")
st.markdown("---")


col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness", 0.0, 500.0, 200.0)
    solids = st.number_input("Solids (TDS)", 0.0, 60000.0, 20000.0)

with col2:
    chloramines = st.number_input("Chloramines", 0.0, 15.0, 7.0)
    sulfate = st.number_input("Sulfate", 0.0, 500.0, 330.0)
    conductivity = st.number_input("Conductivity", 0.0, 800.0, 420.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 14.0)
    trihalomethanes = st.number_input("Trihalomethanes", 0.0, 130.0, 66.0)
    turbidity = st.number_input("Turbidity", 0.0, 7.0, 4.0)

st.markdown("---")


if st.button("🔍 Predict Potability", use_container_width=True):

    payload = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()["result"]

            st.markdown("### Result")

            if "NOT" in result:
                st.error(f"❌ {result}")
            else:
                st.success(f"✅ {result}")

        else:
            st.error("API Error. Please check backend.")

    except:
        st.error("Cannot connect to API. Make sure FastAPI is running.")