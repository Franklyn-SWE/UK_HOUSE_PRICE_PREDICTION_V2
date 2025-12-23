import streamlit as st
import pandas as pd
import dill
import numpy as np
import sklearn
import time
import json
import hashlib
from services.explanation_service import ExplanationService


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

# Initialize OpenAI Explanation Service
@st.cache_resource
def get_explanation_service():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        return ExplanationService(api_key=api_key)
    except Exception as e:
        st.warning(f"OpenAI service unavailable: {e}")
        return None


# In-memory explanation rate limiter and cache (per Streamlit session)
EXPLAIN_LIMIT = 2        # requests
EXPLAIN_PERIOD = 60 * 60  # seconds (1 hour)
EXPLAIN_CACHE_TTL = 60 * 60 * 24  # 24 hours

def _init_explain_state():
    if 'explain_quota' not in st.session_state:
        st.session_state['explain_quota'] = {'count': 0, 'reset': time.time() + EXPLAIN_PERIOD}
    if 'explanation_cache' not in st.session_state:
        st.session_state['explanation_cache'] = {}

def _quota_remaining() -> int:
    q = st.session_state['explain_quota']
    now = time.time()
    if now > q['reset']:
        q['count'] = 0
        q['reset'] = now + EXPLAIN_PERIOD
    return max(0, EXPLAIN_LIMIT - q['count'])

def _consume_quota():
    st.session_state['explain_quota']['count'] += 1

def _get_cached_explanation(cache_key: str):
    cache = st.session_state['explanation_cache']
    entry = cache.get(cache_key)
    if not entry:
        return None
    if time.time() - entry['ts'] > EXPLAIN_CACHE_TTL:
        # expired
        del cache[cache_key]
        return None
    return entry['value']

def _set_cached_explanation(cache_key: str, value: str):
    st.session_state['explanation_cache'][cache_key] = {'value': value, 'ts': time.time()}


def _safe_rerun():
    """Try to force a Streamlit rerun in a way that's compatible across versions.

    Prefer `st.experimental_rerun()` when available; otherwise toggle a query
    parameter which also triggers a rerun.
    """
    try:
        # Newer/older Streamlit versions may or may not expose this helper
        st.experimental_rerun()
    except Exception:
        try:
            # Fallback: change a query param to force a rerun (assign to `st.query_params`)
            # `st.experimental_set_query_params` is deprecated; assign to `st.query_params` instead
            st.query_params = {"_rerun": int(time.time())}
        except Exception:
            # Last-resort: do nothing (UI will update on next interaction)
            return

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
            
            # Store prediction results for explanation
            st.session_state.last_prediction = {
                'input_data': input_df.to_dict('records')[0],
                'predicted_price': prediction_value
            }
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

st.divider()
st.subheader("🧠 AI Explanation (optional)")

# Show quota / usage widget (read live from session_state)
_init_explain_state()
quota = st.session_state['explain_quota']
count_used = int(quota.get('count', 0))
quota_reset_in = int(quota['reset'] - time.time())
minutes = quota_reset_in // 60
seconds = quota_reset_in % 60
# Use placeholders so we can update the displayed quota in-place after generation
col_status, col_action = st.columns([3,1])
quota_area = col_status.empty()
caption_area = col_status.empty()
progress_area = col_action.empty()
quota_area.markdown(f"**Explanation quota:** {count_used}/{EXPLAIN_LIMIT} used")
caption_area.caption(f"Window resets in {minutes}m {seconds}s")
pct = count_used / max(EXPLAIN_LIMIT, 1)
progress_area.progress(min(max(pct, 0.0), 1.0))

if st.button("Explain this prediction"):
    if 'last_prediction' not in st.session_state:
        st.warning("⚠️ Please make a prediction first before requesting an explanation.")
    else:
        try:
            explanation_service = get_explanation_service()
            if explanation_service:
                with st.spinner("Generating explanation..."):
                    _init_explain_state()
                    prediction_info = st.session_state.last_prediction

                    # Preprocess input_data to replace property type codes
                    input_data_for_expl = prediction_info['input_data'].copy()
                    property_type_map = {
                        'D': 'detached house',
                        'S': 'semi-detached house',
                        'T': 'terraced house',
                        'F': 'flat',
                        'O': 'other property type'
                    }
                    if 'property_type' in input_data_for_expl:
                        code = input_data_for_expl['property_type']
                        input_data_for_expl['property_type'] = property_type_map.get(code, input_data_for_expl['property_type'])

                    # Build cache key from input and price
                    cache_key = hashlib.sha256(json.dumps({
                        'input': input_data_for_expl,
                        'price': prediction_info['predicted_price']
                    }, sort_keys=True).encode()).hexdigest()

                    # Check cache
                    cached = _get_cached_explanation(cache_key)
                    if cached:
                        st.session_state['last_explanation'] = cached
                    else:
                        # Check quota
                        remaining = _quota_remaining()
                        if remaining <= 0:
                            # Compute time until reset and inform the user
                            q = st.session_state['explain_quota']
                            reset_in = int(max(0, q['reset'] - time.time()))
                            mins = reset_in // 60
                            secs = reset_in % 60
                            warning_msg = f"Rate limit reached: you've used all {EXPLAIN_LIMIT} explanations. Please wait {mins}m {secs}s for the quota to reset."
                            st.warning(warning_msg)
                            st.session_state['last_explanation'] = warning_msg
                        else:
                            _consume_quota()
                            explanation = explanation_service.generate_explanation(
                                input_data_for_expl,
                                prediction_info['predicted_price']
                            )
                            _set_cached_explanation(cache_key, explanation)
                            st.session_state['last_explanation'] = explanation
                            # Update placeholders so the quota UI reflects the consumed request
                            try:
                                # Recompute values
                                quota = st.session_state['explain_quota']
                                used = int(quota.get('count', 0))
                                reset_in = int(quota['reset'] - time.time())
                                mins = reset_in // 60
                                secs = reset_in % 60
                                quota_area.markdown(f"**Explanation quota:** {used}/{EXPLAIN_LIMIT} used")
                                caption_area.caption(f"Window resets in {mins}m {secs}s")
                                progress_area.progress(min(max(used / max(EXPLAIN_LIMIT, 1), 0.0), 1.0))
                            except Exception:
                                # If placeholders are not available for any reason, skip live update
                                pass
                # show whatever explanation is stored (cached or generated)
                if 'last_explanation' in st.session_state:
                    st.caption(f"Explanation requests remaining this window: {_quota_remaining()}")
                    st.write(st.session_state['last_explanation'])
            else:
                st.warning("AI explanation service is currently unavailable.")
        except Exception as e:
            st.warning("AI explanation is currently unavailable. Please try again later.")
            st.exception(e)

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
