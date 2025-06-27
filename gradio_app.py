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
