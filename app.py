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
    with open('full_pipeline_and_model.pkl', 'rb') as f:
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
    return df


def predict_with_artifact(input_df: pd.DataFrame, artifact: dict) -> float:
    feature_config = artifact["feature_config"]
    model = artifact["model"]
    target_transform = feature_config.get("target_transform") or artifact.get("target_transform")
    df_features = add_date_features(input_df)
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
