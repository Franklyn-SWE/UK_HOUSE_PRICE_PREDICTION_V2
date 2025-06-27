'''
# Load pipeline
with open('full_pipeline_and_model.pkl', 'rb') as f:
    pipeline = dill.load(f)

date_extractor = pipeline['date_extractor']
target_encoder = pipeline['target_encoder']
preprocessor = pipeline['preprocessor']
model = pipeline['model']

def predict(date, property_type, new_build, freehold, town, district, county, street, locality, postcode):
    input_df = pd.DataFrame([{
        'date': date,
        'property_type': property_type,
        'new_build': new_build,
        'freehold': freehold,
        'town': town,
        'district': district,
        'county': county,
        'street': street,
        'locality': locality,
        'postcode': postcode
    }])
    df_date = date_extractor.transform(input_df)
    df_te = target_encoder.transform(df_date[target_encoder.cols])
    X_input = pd.concat([
        df_date.drop(columns=target_encoder.cols + ['date']),
        df_te
    ], axis=1)
    X_preprocessed = preprocessor.transform(X_input)
    prediction = model.predict(X_preprocessed)
    return f"£{prediction[0]:,.2f}"

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Date (YYYY-MM-DD)"),
        gr.Dropdown(['D', 'S', 'T', 'F', 'O'], label="Property Type"),
        gr.Slider(0, 1, step=1, label="New Build"),
        gr.Slider(0, 1, step=1, label="Freehold"),
        gr.Textbox(label="Town"),
        gr.Textbox(label="District"),
        gr.Textbox(label="County"),
        gr.Textbox(label="Street"),
        gr.Textbox(label="Locality"),
        gr.Textbox(label="Postcode"),
    ],
    outputs="text"
)

iface.launch()

=================================================================================================================================
import dill
import pandas as pd
import gradio as gr
from datetime import datetime

# Load pipeline components
with open('full_pipeline_and_model.pkl', 'rb') as f:
    pipeline = dill.load(f)

date_extractor = pipeline['date_extractor']
target_encoder = pipeline['target_encoder']
preprocessor = pipeline['preprocessor']
model = pipeline['model']

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
        'new_build': int(new_build),  # gr.Checkbox returns bool, convert to int
        'freehold': int(freehold),
        'town': town,
        'district': district,
        'county': county,
        'street': street,
        'locality': locality,
        'postcode': postcode
    }])

    # Transformations & prediction pipeline
    transformed_date = date_extractor.transform(input_data)
    target_encoded = target_encoder.transform(transformed_date[target_encoder.cols])
    model_input = pd.concat([
        transformed_date.drop(columns=target_encoder.cols + ['date']),
        target_encoded
    ], axis=1)
    processed_input = preprocessor.transform(model_input)
    prediction = model.predict(processed_input)

    return f"Estimated House Price: £{prediction[0]:,.2f}"

iface = gr.Interface(
    fn=predict_house_price,
    inputs=[
        gr.Textbox(label="Date (YYYY-MM-DD)", placeholder="e.g., 2023-08-15"),
        gr.Dropdown(['D', 'S', 'T', 'F', 'O'], label="Property Type", info="D=Detached, S=Semi-detached, T=Terraced, F=Flat, O=Other"),
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



=============================================================================
STREAMLIT APP CODES 
=============================================================================
import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor

#import streamlit as st
import pandas as pd
import dill

@st.cache_resource
def load_pipeline():
    with open('full_pipeline_and_model.pkl', 'rb') as f:
        pipeline = dill.load(f)
    return pipeline

pipeline = load_pipeline()
date_extractor = pipeline['date_extractor']
target_encoder = pipeline['target_encoder']
preprocessor = pipeline['preprocessor']
model = pipeline['model']

st.title("UK House Price Prediction")

# User inputs
date = st.date_input("Date of Sale")
property_type = st.selectbox("Property Type", ['D', 'S', 'T', 'F', 'O'])
new_build = st.selectbox("New Build", [0, 1])
freehold = st.selectbox("Freehold", [0, 1])
town = st.text_input("Town")
district = st.text_input("District")
county = st.text_input("County")
street = st.text_input("Street")
locality = st.text_input("Locality")
postcode = st.text_input("Postcode")

if st.button("Predict Price"):
    # Prepare input dataframe
    input_df = pd.DataFrame([{
        'date': date.strftime("%Y-%m-%d"),
        'property_type': property_type,
        'new_build': new_build,
        'freehold': freehold,
        'town': town,
        'district': district,
        'county': county,
        'street': street,
        'locality': locality,
        'postcode': postcode
    }])

    # Run the full pipeline inference
    try:
        df_date = date_extractor.transform(input_df)
        df_te = target_encoder.transform(df_date[target_encoder.cols])
        X_input = pd.concat([
            df_date.drop(columns=target_encoder.cols + ['date']),
            df_te
        ], axis=1)
        X_preprocessed = preprocessor.transform(X_input)
        prediction = model.predict(X_preprocessed)

        st.success(f"Predicted House Price: £{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

============================================================================
import streamlit as st
import pandas as pd
import dill

DATA_PATH = r"D:\UK_HOUSE_PRICE_PREDICTION_PROJECT\data\UK_House_Price_Prediction_dataset_2015_to_2024.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_pipeline():
    with open('full_pipeline_and_model.pkl', 'rb') as f:
        return dill.load(f)

df = load_data()
pipeline = load_pipeline()

date_extractor = pipeline['date_extractor']
target_encoder = pipeline['target_encoder']
preprocessor = pipeline['preprocessor']
model = pipeline['model']

PROPERTY_TYPES = {
    'D': 'Detached',
    'S': 'Semi-Detached',
    'T': 'Terraced',
    'F': 'Flat',
    'O': 'Other'
}

st.title("🏠 UK House Price Prediction")

# Step 1: Town selection
towns = sorted(df['town'].dropna().unique())
town = st.selectbox("Town", towns)

# Step 2: Filter districts and counties based on town
filtered_df = df[df['town'] == town]
districts = sorted(filtered_df['district'].dropna().unique())
counties = sorted(filtered_df['county'].dropna().unique())

# Step 3: Show district and county dropdowns with filtered values
district = st.selectbox("District", districts)
county = st.selectbox("County", counties)

# Step 4: Other inputs
date = st.date_input("Date of Sale", help="Select the date the property was sold")
property_type = st.selectbox("Property Type", options=list(PROPERTY_TYPES.keys()), format_func=lambda x: PROPERTY_TYPES[x])
new_build = st.checkbox("New Build")
freehold = st.checkbox("Freehold")
street = st.text_input("Street", placeholder="e.g., Deansgate")
locality = st.text_input("Locality", placeholder="e.g., City Centre")
postcode = st.text_input("Postcode", placeholder="e.g., M3 4LX")

# Step 5: Prediction button
if st.button("Predict Price"):
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
                'town': town,
                'district': district,
                'county': county,
                'street': street,
                'locality': locality,
                'postcode': postcode
            }])

            df_date = date_extractor.transform(input_df)
            df_te = target_encoder.transform(df_date[target_encoder.cols])
            X_input = pd.concat([
                df_date.drop(columns=target_encoder.cols + ['date']),
                df_te
            ], axis=1)
            X_preprocessed = preprocessor.transform(X_input)
            prediction = model.predict(X_preprocessed)

            st.success(f"🏷️ Predicted House Price: **£{prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
            
=============================================================================
import streamlit as st
import pandas as pd
import dill

DATA_PATH = r"D:\UK_HOUSE_PRICE_PREDICTION_PROJECT\data\UK_House_Price_Prediction_dataset_2015_to_2024.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_pipeline():
    with open('full_pipeline_and_model.pkl', 'rb') as f:
        return dill.load(f)

df = load_data()
pipeline = load_pipeline()

date_extractor = pipeline['date_extractor']
target_encoder = pipeline['target_encoder']
preprocessor = pipeline['preprocessor']
model = pipeline['model']

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
                'postcode': postcode
            }])

            df_date = date_extractor.transform(input_df)
            df_te = target_encoder.transform(df_date[target_encoder.cols])
            X_input = pd.concat([
                df_date.drop(columns=target_encoder.cols + ['date']),
                df_te
            ], axis=1)
            X_preprocessed = preprocessor.transform(X_input)
            prediction = model.predict(X_preprocessed)

            st.success(f"🏷️ Predicted House Price: **£{prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")


'''