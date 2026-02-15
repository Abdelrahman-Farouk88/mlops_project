import streamlit as st
import pandas as pd
import pickle
import mlflow
import dagshub
import os

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="centered"
)

st.title("💧 Water Potability Prediction")
st.markdown("Predict whether water is **safe to drink** based on water quality parameters.")
st.markdown("---")


@st.cache_resource
def load_model():
    """Load the production model from DagsHub MLflow registry."""
    dagshub.init(repo_owner='Abdelrahman-Farouk88', repo_name='mlops_project', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Abdelrahman-Farouk88/mlops_project.mlflow")

    client = mlflow.tracking.MlflowClient()
    model_name = "Best Model"

    versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        st.error("No model found in Production stage.")
        return None

    run_id = prod_versions[0].run_id
    version = prod_versions[0].version

    artifact_path = client.download_artifacts(run_id, "model.pkl")
    with open(artifact_path, "rb") as f:
        model = pickle.load(f)

    return model, version


model_data = load_model()

if model_data:
    model, model_version = model_data

    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Model:** Best Model")
    st.sidebar.write(f"**Version:** {model_version}")
    st.sidebar.write(f"**Stage:** Production")
    st.sidebar.write(f"**Algorithm:** Random Forest")

    st.subheader("Enter Water Quality Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1,
                             help="pH of water (0-14)")
        hardness = st.number_input("Hardness", min_value=0.0, max_value=500.0, value=200.0, step=1.0,
                                   help="Capacity of water to precipitate soap (mg/L)")
        solids = st.number_input("Solids (TDS)", min_value=0.0, max_value=60000.0, value=20000.0, step=100.0,
                                 help="Total dissolved solids (ppm)")

    with col2:
        chloramines = st.number_input("Chloramines", min_value=0.0, max_value=15.0, value=7.0, step=0.1,
                                      help="Amount of chloramines (ppm)")
        sulfate = st.number_input("Sulfate", min_value=0.0, max_value=500.0, value=330.0, step=1.0,
                                  help="Amount of sulfate (mg/L)")
        conductivity = st.number_input("Conductivity", min_value=0.0, max_value=800.0, value=420.0, step=1.0,
                                       help="Electrical conductivity (μS/cm)")

    with col3:
        organic_carbon = st.number_input("Organic Carbon", min_value=0.0, max_value=30.0, value=14.0, step=0.1,
                                         help="Amount of organic carbon (ppm)")
        trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, max_value=130.0, value=66.0, step=0.1,
                                          help="Amount of trihalomethanes (μg/L)")
        turbidity = st.number_input("Turbidity", min_value=0.0, max_value=7.0, value=4.0, step=0.1,
                                    help="Measure of light emitting property (NTU)")

    st.markdown("---")

    if st.button("🔍 Predict Potability", use_container_width=True):
        input_data = pd.DataFrame({
            'ph': [ph],
            'Hardness': [hardness],
            'Solids': [solids],
            'Chloramines': [chloramines],
            'Sulfate': [sulfate],
            'Conductivity': [conductivity],
            'Organic_carbon': [organic_carbon],
            'Trihalomethanes': [trihalomethanes],
            'Turbidity': [turbidity]
        })

        prediction = model.predict(input_data)

        st.markdown("### Result")
        if prediction[0] == 1:
            st.success("✅ **Water is Potable** — Safe to drink!")
        else:
            st.error("❌ **Water is Not Potable** — Not safe to drink!")

        st.markdown("### Input Summary")
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
