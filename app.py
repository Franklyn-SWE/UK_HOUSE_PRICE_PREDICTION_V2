import streamlit as st
import pandas as pd
import dill
import numpy as np
import sklearn

# Path to the dataset
DATA_PATH = "data/UK_House_Price_Prediction_dataset_2015_to_2024.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_pipeline():
    import os
    import sys
    import subprocess
    
    model_path = 'full_pipeline_and_model.pkl'
    
    # If model doesn't exist, train it
    if not os.path.exists(model_path):
        st.warning("⏳ Model not found. Training now (this takes ~5-6 minutes)...")
        st.info("Training UK House Price Prediction Model v1.1...")
        
        try:
            # Use sys.executable to ensure we use the same Python interpreter
            result = subprocess.run(
                [sys.executable, 'training/train.py'],
                capture_output=True,
                text=True,
                check=True,
                env=os.environ.copy()  # Pass the environment variables
            )
            st.success("✅ Model trained successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Model training failed!")
            st.error(f"Error output: {e.stderr[:500]}")  # Show first 500 chars
            raise
    
    # Load the model
    with open(model_path, 'rb') as f:
        return dill.load(f)

# Load data and pipeline
df = load_data()
pipeline = load_pipeline()

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_quarter"] = df["date"].dt.quarter
    df["sale_dayofweek"] = df["date"].dt.dayofweek
    df["sale_is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["years_since_2015"] = df["sale_year"] - 2015  # Linear time trend
    return df


def add_postcode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hierarchical postcode information"""
    df = df.copy()
    df["postcode"] = df["postcode"].astype(str).str.upper().str.strip()
    
    # Extract postcode area (e.g., "SW1A" from "SW1A 1AA")
    df["postcode_area"] = df["postcode"].str.split().str[0]
    
    # Extract postcode district (e.g., "SW" from "SW1A 1AA")
    df["postcode_district"] = df["postcode"].str.extract(r'^([A-Z]+)', expand=False)
    
    return df


def add_target_encoding_inference(input_df: pd.DataFrame, encoding_map: dict, 
                                  train_mean: float) -> pd.DataFrame:
    """Add target-encoded features during inference"""
    df = input_df.copy()
    
    # CRITICAL: encoding_map already contains log-transformed values
    # Apply encoding, using log-transformed train mean as fallback
    train_mean_log = np.log1p(train_mean)
    
    # NOTE: postcode_mean_price removed to prevent overfitting
    df["town_mean_price"] = df["town"].map(
        encoding_map.get("town", {})
    ).fillna(train_mean_log)
    
    df["district_mean_price"] = df["district"].map(
        encoding_map.get("district", {})
    ).fillna(train_mean_log)
    
    df["county_mean_price"] = df["county"].map(
        encoding_map.get("county", {})
    ).fillna(train_mean_log)
    
    df["property_type_mean_price"] = df["property_type"].map(
        encoding_map.get("property_type", {})
    ).fillna(train_mean_log)
    
    return df


def predict_with_artifact(input_df: pd.DataFrame, artifact: dict) -> float:
    feature_config = artifact["feature_config"]
    model = artifact["model"]
    target_transform = feature_config.get("target_transform") or artifact.get("target_transform")
    encoding_map = artifact.get("encoding_map", {})
    
    # Add all feature transformations
    df_features = add_date_features(input_df)
    df_features = add_postcode_features(df_features)
    
    # Add target encoding if available
    if encoding_map:
        train_mean = 240000  # Fallback value
        df_features = add_target_encoding_inference(df_features, encoding_map, train_mean)
    
    # Prepare text features
    for column in ["street", "locality", "town", "district", "county"]:
        if column in df_features.columns:
            df_features[column] = df_features[column].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    
    X_input = df_features[feature_config["feature_columns"]]
    prediction_log = model.predict(X_input)
    
    if target_transform == "log1p":
        prediction = np.expm1(prediction_log)
    else:
        prediction = prediction_log
    return float(prediction[0])


date_extractor = pipeline.get('date_extractor')
target_encoder = pipeline.get('target_encoder')
preprocessor = pipeline.get('preprocessor')
model = pipeline.get('model')

PROPERTY_TYPES = {
    'D': 'Detached',
    'S': 'Semi-Detached',
    'T': 'Terraced',
    'F': 'Flat',
    'O': 'Other'
}

st.title("🏠 UK House Price Prediction")

# Get list of towns
towns = sorted(df['town'].dropna().unique())

# Track selected town in session_state
if 'selected_town' not in st.session_state:
    st.session_state.selected_town = towns[0]

# Town selection (outside form so it updates dynamically)
selected_town = st.selectbox("🏙️ Select Town", towns, index=towns.index(st.session_state.selected_town))
st.session_state.selected_town = selected_town

# Filter district and county based on selected town
filtered_df = df[df['town'] == selected_town]
districts = sorted(filtered_df['district'].dropna().unique())
counties = sorted(filtered_df['county'].dropna().unique())

# Default values for district and county
default_district = districts[0] if districts else ""
default_county = counties[0] if counties else ""

# 📝 Prediction Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("📅 Date of Sale")
        property_type = st.selectbox("🏠 Property Type", list(PROPERTY_TYPES.keys()), format_func=lambda x: PROPERTY_TYPES[x])
        new_build = st.checkbox("🏗️ New Build")
        freehold = st.checkbox("📜 Freehold")
        district = st.selectbox("📌 District", districts, index=0 if default_district in districts else 0)
    with col2:
        county = st.selectbox("📍 County", counties, index=0 if default_county in counties else 0)
        street = st.text_input("🏘️ Street", placeholder="e.g., Deansgate")
        locality = st.text_input("🏙️ Locality", placeholder="e.g., City Centre")
        postcode = st.text_input("✉️ Postcode", placeholder="e.g., M3 4LX")

    submit = st.form_submit_button("🔮 Predict Price")

# Run prediction after submission
if submit:
    if not street.strip():
        st.error("Please enter the street name.")
    elif not locality.strip():
        st.error("Please enter the locality.")
    elif not postcode.strip():
        st.error("Please enter the postcode.")
    else:
        try:
            input_df = pd.DataFrame([{
                'date': date.strftime("%Y-%m-%d"),
                'property_type': property_type,
                'new_build': int(new_build),
                'freehold': int(freehold),
                'town': selected_town,
                'district': district,
                'county': county,
                'street': street,
                'locality': locality,
                'postcode': postcode.upper().strip()
            }])

            if "feature_config" in pipeline:
                prediction_value = predict_with_artifact(input_df, pipeline)
            else:
                df_date = date_extractor.transform(input_df)
                df_te = target_encoder.transform(df_date[target_encoder.cols])
                X_input = pd.concat([
                    df_date.drop(columns=target_encoder.cols + ['date']),
                    df_te
                ], axis=1)
                X_preprocessed = preprocessor.transform(X_input)
                prediction_value = model.predict(X_preprocessed)[0]
                if pipeline.get("target_transform") == "log1p":
                    prediction_value = np.expm1(prediction_value)

            st.success(f"🏷️ Predicted House Price: **£{prediction_value:,.2f}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# Footer with credits and LinkedIn
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px;'>
        Developed by <strong>Franklyn  Oliha</strong> |
        <strong>Mudia Estate and Tech</strong><br>
        <a href="https://www.linkedin.com/in/franklyn-oliha/" target="_blank">🔗 Connect on LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
