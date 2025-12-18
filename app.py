import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Set page config
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model and feature names
@st.cache_resource
def load_model_and_data():
    model = joblib.load("house_price_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    # Load dataset for visualization
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return model, feature_names, df, data.target

try:
    model, feature_names, df, target = load_model_and_data()
except FileNotFoundError:
    st.error("Model files not found. Please run 'model.py' first.")
    st.stop()

# Title and Description
st.title("🏠 California House Price Prediction")
st.markdown("""
This app predicts the **median house value** in California districts based on various features.
Adjust the parameters in the sidebar to seeing how they affect the price!
""")

st.divider()

# Sidebar - Inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    MedInc = st.sidebar.slider("Median Income (tens of thousands)", 0.5, 15.0, 3.8, 0.1)
    HouseAge = st.sidebar.slider("House Age (years)", 1, 52, 28, 1)
    AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0, 0.1)
    AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0, 0.1)
    Population = st.sidebar.slider("Population", 100, 10000, 1000, 100)
    AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 6.0, 3.0, 0.1)
    Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.6, 0.1)
    Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -119.5, 0.1)
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Your Input Parameters")
    st.dataframe(input_df)

    # Prediction
    st.subheader("Prediction")
    if st.button("Predict Price", type="primary"):
        prediction = model.predict(input_df)
        value = prediction[0] * 100000 # Dataset is in 100k units
        # Contextual comparison
        avg_value = target.mean() * 100000
        diff = value - avg_value
        
        st.metric(
            label="Estimated Median House Value",
            value=f"${value:,.2f}",
            delta=f"{diff:,.2f} vs Avg",
        )
        st.caption(f"The average house value in this dataset is ${avg_value:,.2f}.")

with col2:
    st.subheader("Dataset Insights")
    
    # Simple correlation heatmap
    if st.checkbox("Show Correlation Heatmap", value=True):
        st.write("Correlation of features with Price")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Feature Importance
    if st.checkbox("Show Feature Importance", value=False):
        st.write("What affects prices the most?")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots()
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.caption("Built with Streamlit & Scikit-Learn | Data: California Housing")
