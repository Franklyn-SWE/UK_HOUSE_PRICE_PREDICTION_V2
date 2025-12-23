"""
OpenAI Explanation Service for UK House Price Predictions.

This service generates clear, non-technical explanations
for ML-based house price predictions.
"""

import os
from openai import OpenAI


class ExplanationService:
    def __init__(self, api_key: str | None = None):
        """
        Initialize the ExplanationService.

        Args:
            api_key (str, optional): OpenAI API key.
            If not provided, reads from OPENAI_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your environment or Streamlit secrets."
            )

        self.client = OpenAI(api_key=self.api_key)
        

    def generate_explanation(self, input_data: dict, predicted_price: float) -> str:
        """
        Generate a natural language explanation for a house price prediction.

        Args:
            input_data (dict): Property features used for prediction
            predicted_price (float): Model-predicted house price

        Returns:
            str: Human-readable explanation
        """
        # Map property type codes to full names
        property_type_map = {
            'D': 'detached house',
            'S': 'semi-detached house',
            'T': 'terraced house',
            'F': 'flat',
            'O': 'other property type'
        }
        property_type_code = input_data.get("property_type")
        property_type_name = property_type_map.get(property_type_code, "property")
        # If the code is not recognized, just use 'property' (never the code itself)

        tenure_type = "freehold" if input_data.get("freehold") else "leasehold"
        new_build_status = "new build" if input_data.get("new_build") else "existing property"
        new_build_clause = " and is a new build" if input_data.get("new_build") else ""

        town = input_data.get("town", "[unknown town]")
        postcode = input_data.get("postcode", "[unknown postcode]")
        clean_property_description = f"{property_type_name} ({new_build_status}, {tenure_type})"

        prompt = f"""
You are a UK property market analyst.

Property details:
- Location: {town}, Postcode: {postcode}
- Property: {clean_property_description}
- Predicted price: £{predicted_price:,.0f}

IMPORTANT:
This is a {property_type_name}. Write naturally about this {property_type_name}.
Do NOT mention any codes, letters, abbreviations, or internal labels. Never say things like 'indicated by F' or '(F)'.
Always describe it simply as a {property_type_name}.

Format your response exactly as follows:

**Explanation:**
Write 2–3 sentences explaining why this {property_type_name} in {town}
is valued at £{predicted_price:,.0f}. Discuss local demand, location characteristics,
and why this price is appropriate for a {property_type_name} in this area.

**Key Influencing Factors:**
- Location: Characteristics of {town}, including accessibility and local amenities
- Property features: This {property_type_name} is held under {tenure_type} tenure{new_build_clause}, which influences its appeal and market value.
- Market conditions: Current demand for similar properties in the area

**Confidence Note:**
Write one concise sentence explaining that this estimate is based on available data
and recent market trends. Avoid disclaimers, uncertainty language, or dates.

Use natural, professional, non-technical language throughout.
Refer to the property as "this {property_type_name}" or "the {property_type_name}".
Never use codes, letters in parentheses, or abbreviations.
""".strip()

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a UK property analyst. Always describe properties using full names (e.g., 'flat', 'apartment', 'detached house') never codes or abbreviations."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()