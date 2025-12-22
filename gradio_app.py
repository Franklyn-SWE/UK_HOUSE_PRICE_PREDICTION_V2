import dill
import pandas as pd
import numpy as np
import gradio as gr
from datetime import datetime

# Load pipeline components
with open('full_pipeline_and_model.pkl', 'rb') as f:
    pipeline = dill.load(f)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_quarter"] = df["date"].dt.quarter
    df["sale_dayofweek"] = df["date"].dt.dayofweek
    df["sale_is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["years_since_2015"] = df["sale_year"] - 2015
    return df


def add_postcode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["postcode"] = df["postcode"].astype(str).str.upper().str.strip()
    df["postcode_area"] = df["postcode"].str.split().str[0]
    df["postcode_district"] = df["postcode"].str.extract(r'^([A-Z]+)', expand=False)
    return df


def add_target_encoding_inference(input_df: pd.DataFrame, encoding_map: dict, 
                                  train_mean: float) -> pd.DataFrame:
    df = input_df.copy()
    
    # CRITICAL: encoding_map already contains log-transformed values
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

date_extractor = pipeline['date_extractor']

def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True, ""
    except ValueError:
        return False, "Date must be in YYYY-MM-DD format."

def validate_required(text, field_name):
    if not text or text.strip() == "":
        return False, f"{field_name} cannot be empty."
    return True, ""

def predict_house_price(date, property_type, new_build, freehold, town, district, county, street, locality, postcode):
    # Validate inputs
    valid, msg = validate_date(date)
    if not valid:
        return f"Error: {msg}"
    
    for field_name, field_value in [
        ("Town", town),
        ("District", district),
        ("County", county),
        ("Street", street),
        ("Locality", locality),
        ("Postcode", postcode)
    ]:
        valid, msg = validate_required(field_value, field_name)
        if not valid:
            return f"Error: {msg}"
    
    # Create DataFrame from inputs
    input_data = pd.DataFrame([{
        'date': date,
        'property_type': property_type,
        'new_build': int(new_build),
        'freehold': int(freehold),
        'town': town,
        'district': district,
        'county': county,
        'street': street,
        'locality': locality,
        'postcode': postcode
    }])

    # Check if using new artifact format
    if "feature_config" in pipeline:
        feature_config = pipeline["feature_config"]
        model = pipeline["model"]
        target_transform = feature_config.get("target_transform") or pipeline.get("target_transform")
        encoding_map = pipeline.get("encoding_map", {})
        
        # Add all feature transformations
        df_features = add_date_features(input_data)
        df_features = add_postcode_features(df_features)
        
        # Add target encoding if available
        if encoding_map:
            train_mean = 240000
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
        prediction_value = float(prediction[0])
    else:
        # Old pipeline format
        date_extractor = pipeline['date_extractor']
        target_encoder = pipeline['target_encoder']
        preprocessor = pipeline['preprocessor']
        model = pipeline['model']
        
        transformed_date = date_extractor.transform(input_data)
        target_encoded = target_encoder.transform(transformed_date[target_encoder.cols])
        model_input = pd.concat([
            transformed_date.drop(columns=target_encoder.cols + ['date']),
            target_encoded
        ], axis=1)
        processed_input = preprocessor.transform(model_input)
        prediction = model.predict(processed_input)
        prediction_value = prediction[0]

    return f"Estimated House Price: £{prediction_value:,.2f}"

iface = gr.Interface(
    fn=predict_house_price,
    inputs=[
        gr.Textbox(label="Date (YYYY-MM-DD)", placeholder="e.g., 2023-08-15"),
        gr.Dropdown(['Detached', 'Semi-detached', 'Terraced', 'Flat', 'Other'], label="Property Type", info="D=Detached, S=Semi-detached, T=Terraced, F=Flat, O=Other"),
        gr.Checkbox(label="New Build"),
        gr.Checkbox(label="Freehold"),
        gr.Textbox(label="Town", placeholder="e.g., Manchester"),
        gr.Textbox(label="District", placeholder="e.g., Manchester District"),
        gr.Textbox(label="County", placeholder="e.g., Greater Manchester"),
        gr.Textbox(label="Street", placeholder="e.g., Deansgate"),
        gr.Textbox(label="Locality", placeholder="e.g., City Centre"),
        gr.Textbox(label="Postcode", placeholder="e.g., M3 4LX"),
    ],
    outputs=gr.Textbox(label="Predicted House Price or Error"),
    title="UK House Price Prediction",
    description="Enter property details to get an estimated house price. Please fill in all required fields."
)

if __name__ == "__main__":
    iface.launch()
