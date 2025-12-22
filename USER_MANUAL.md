# UK House Price Prediction - User Manual 📖

## Table of Contents
1. [Getting Started](#getting-started)
2. [Accessing the Application](#accessing-the-application)
3. [Understanding the Interface](#understanding-the-interface)
4. [Making a Prediction](#making-a-prediction)
5. [Input Fields Explained](#input-fields-explained)
6. [Interpreting Results](#interpreting-results)
7. [Tips for Accurate Predictions](#tips-for-accurate-predictions)
8. [Common Issues & Solutions](#common-issues--solutions)
9. [Frequently Asked Questions](#frequently-asked-questions)

---

## Getting Started

The UK House Price Prediction app uses machine learning to estimate property prices based on historical UK house sale data from 2015-2024. The model analyzes over 90,000 transactions to provide price predictions.

### What You Need
- Web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- Property details (address, type, features)

---

## Accessing the Application

### Online (Recommended)
Visit the live app at: **[Your Streamlit App URL]**

### Local Installation
If running locally:
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

---

## Understanding the Interface

The app has a clean, user-friendly interface with three main sections:

### 1. Header
- Title and version information
- Brief description of the model

### 2. Input Panel (Left Sidebar or Top Section)
- Form fields for property details
- All fields are required for prediction

### 3. Results Panel (Main Area)
- Predicted price display
- Confidence metrics
- Feature importance visualization

---

## Making a Prediction

### Step-by-Step Guide

#### Step 1: Enter Property Location
```
📍 LOCATION DETAILS
```

**Postcode**
- Format: `SW1A 1AA` or `LE17 5AP`
- Enter the full UK postcode
- The app automatically extracts area and district codes
- Example: `M1 1AE` (Manchester city center)

**Street**
- Enter the street name
- Example: `HIGH STREET` or `PARK ROAD`
- Case insensitive

**Locality**
- Specific area or neighborhood
- Example: `CITY CENTRE`, `TOWN CENTRE`
- Enter `UNKNOWN` if not applicable

**Town**
- The town or city name
- Example: `LEICESTER`, `MANCHESTER`, `LONDON`

**District**
- Administrative district
- Example: `HARBOROUGH`, `MANCHESTER`, `WESTMINSTER`

**County**
- County name
- Example: `LEICESTERSHIRE`, `GREATER MANCHESTER`, `LONDON`

#### Step 2: Specify Property Type
```
🏠 PROPERTY DETAILS
```

**Property Type** (Dropdown)
Select one:
- **D** - Detached House (standalone, no shared walls)
- **S** - Semi-Detached House (shares one wall with neighbor)
- **T** - Terraced House (shares walls on both sides)
- **F** - Flat/Apartment (part of larger building)
- **O** - Other (commercial, land, unusual properties)

#### Step 3: Set Additional Features
```
✨ FEATURES
```

**New Build** (Yes/No)
- **Yes**: Property is newly constructed
- **No**: Previously owned property (resale)

**Freehold** (Yes/No)
- **Yes**: You own the property and land
- **No**: Leasehold - you lease the property for a fixed term

#### Step 4: Select Sale Date
```
📅 SALE DATE
```

**Date**
- Use the date picker or enter manually
- Format: `YYYY-MM-DD`
- Affects seasonal pricing patterns
- Default: Today's date

#### Step 5: Get Prediction

Click the **"Predict Price"** button

---

## Input Fields Explained

### Why Each Field Matters

| Field | Impact on Price | Example Values |
|-------|-----------------|----------------|
| **Postcode** | HIGH - Determines location premium | SW1A 1AA (London center) |
| **Town** | HIGH - Major price factor | LONDON vs LEICESTER |
| **Property Type** | VERY HIGH - 23% of prediction | D (detached) > S > T > F |
| **District** | MEDIUM - Regional variations | HARBOROUGH vs OADBY |
| **County** | MEDIUM - Broad geographic trends | GREATER LONDON vs LEICESTERSHIRE |
| **New Build** | MEDIUM - 10-30% premium | Yes = Higher price |
| **Freehold** | LOW-MEDIUM - Ownership type | Freehold > Leasehold |
| **Date** | LOW - Seasonal effects | Summer > Winter |
| **Street** | LOW - Specific location | MAIN STREET vs SIDE ROAD |
| **Locality** | LOW-MEDIUM - Neighborhood | CITY CENTRE vs SUBURBS |

### Feature Importance Ranking
1. 🥇 **Property Type** (22.93%)
2. 🥈 **Town Mean Price** (12.83%)
3. 🥉 **Property Type Mean Price** (9.93%)
4. **Locality** (6.68%)
5. **Postcode Area** (6.40%)

---

## Interpreting Results

### Predicted Price Display

```
🏠 Predicted House Price
£325,000
```

This is the model's estimate based on:
- Historical sales data (2015-2024)
- 90,000+ similar transactions
- Advanced machine learning algorithm

### Understanding the Prediction

**Accuracy Metrics:**
- **Average Error:** ±£101,505 (MAE)
- **Typical Error:** ±£53,727 (Median AE)
- **Percentage Error:** ±33.8% (MAPE)
- **R² Score:** 0.247 (24.7% variance explained)

### What the Numbers Mean

**If the predicted price is £300,000:**

| Range | Likelihood | Actual Price Could Be |
|-------|------------|----------------------|
| Most Likely | 50% | £246,273 - £353,727 |
| Very Likely | 68% | £198,495 - £401,505 |
| Possible | 95% | £100,000 - £500,000 |

**Confidence Levels:**
- ✅ **High Confidence**: Properties similar to training data
- ⚠️ **Medium Confidence**: Some unusual features
- ❌ **Low Confidence**: Rare property types or locations

### Additional Information Displayed

**Feature Contributions:**
- Shows which features most influenced the prediction
- Helps understand the price breakdown

**Similar Properties:**
- Average prices in the same area
- Comparable recent sales

---

## Tips for Accurate Predictions

### ✅ Do's

1. **Use Accurate Postcodes**
   - Full postcode (e.g., `LE17 5AP`) is better than partial
   - Postcode determines 15%+ of the price

2. **Be Specific with Location**
   - Use official town/district names
   - Match spelling to Land Registry data

3. **Select Correct Property Type**
   - This is the most important feature (23%)
   - Double-check your selection

4. **Update Regularly**
   - Property markets change
   - Recheck predictions every 3-6 months

5. **Use Current Date**
   - For valuation purposes
   - Historical dates for past sale estimates

### ❌ Don'ts

1. **Don't Use Partial Information**
   - All fields affect accuracy
   - "UNKNOWN" should be last resort

2. **Don't Trust 100%**
   - Model has ±34% error on average
   - Use as guidance, not absolute truth

3. **Don't Predict for Unique Properties**
   - Model trained on typical properties
   - Castles, mansions, commercial = unreliable

4. **Don't Ignore Context**
   - Recent renovations not captured
   - Local developments not in data
   - Economic changes may affect prices

---

## Common Issues & Solutions

### Problem: "Model file not found"
**Solution:** 
- Wait 5-6 minutes for automatic training
- Refresh the page after training completes
- Contact support if persists

### Problem: "Invalid postcode format"
**Solution:**
- Use format: `XX## #XX` (e.g., `SW1A 1AA`)
- Include the space
- Check for typos

### Problem: "Prediction seems too high/low"
**Possible Causes:**
- Rare property type for the area
- Recent market changes not in training data
- Typo in location fields

**What to Do:**
- Verify all inputs are correct
- Compare with recent sales in area
- Consider the ±34% margin of error

### Problem: "App is slow"
**Solutions:**
- First load takes 5-6 min (training)
- Subsequent loads are fast (cached)
- Check internet connection
- Try different browser

### Problem: "Date picker not working"
**Solutions:**
- Type date manually: `2024-12-22`
- Use format: `YYYY-MM-DD`
- Clear browser cache

---

## Frequently Asked Questions

### General Questions

**Q: How accurate is the model?**
A: The model has an average error of ±34% (MAPE). For a £300k property, expect predictions within £200k-£400k range. It's most accurate for typical properties in well-represented areas.

**Q: What data is the model trained on?**
A: 90,000+ UK house sales from 2015-2024, from the Land Registry. It includes all regions and property types.

**Q: Can I predict prices for Scotland/Wales/Northern Ireland?**
A: If those areas are in the training data, yes. The model works best for England where most data exists.

**Q: How often is the model updated?**
A: Currently using data up to 2024. Model retraining happens periodically as new data becomes available.

### Technical Questions

**Q: What algorithm does it use?**
A: CatBoost Gradient Boosting with 2000 iterations, depth 8, and advanced feature engineering including target encoding and postcode hierarchy.

**Q: Why is my prediction different from Zoopla/Rightmove?**
A: Different models, data sources, and methodologies. Our model uses only historical sales data, while others may incorporate listings, property characteristics, and agent estimates.

**Q: Can I get more details about a prediction?**
A: Currently shows feature importance. Future versions may include:
- Confidence intervals
- Comparable properties
- Price trend analysis

**Q: Does the model consider property size?**
A: Not yet. This is a known limitation. The model uses location, type, and features but not bedrooms, bathrooms, or square footage.

### Usage Questions

**Q: Can I use this for commercial purposes?**
A: This is a demonstration app. For commercial property valuations, consult professional surveyors and estate agents.

**Q: Can I make multiple predictions?**
A: Yes! No limit on predictions. Each prediction is independent.

**Q: Can I export/save results?**
A: Currently no export feature. Take a screenshot or note the prediction. Future versions may add export functionality.

**Q: Is my data stored?**
A: No. Predictions are not stored. Your input data is only used for the current prediction and is not saved.

---

## Example Use Cases

### Use Case 1: Buying a House
```
Scenario: Considering a 3-bed semi-detached in Leicester

Input:
- Postcode: LE2 3BY
- Property Type: S (Semi-Detached)
- Town: LEICESTER
- District: LEICESTER
- County: LEICESTERSHIRE
- New Build: No
- Freehold: Yes
- Date: 2024-12-22

Prediction: £280,000
Range: £226,000 - £334,000

Action: Compare with asking price (£295k)
Verdict: Asking price within reasonable range
```

### Use Case 2: Selling Your Property
```
Scenario: Setting asking price for flat in Manchester

Input:
- Postcode: M1 1AE
- Property Type: F (Flat)
- Town: MANCHESTER
- District: MANCHESTER
- County: GREATER MANCHESTER
- New Build: No
- Freehold: No (Leasehold)
- Date: 2024-12-22

Prediction: £195,000
Range: £161,000 - £229,000

Action: Compare with estate agent valuation
Verdict: Start at £210k (upper range)
```

### Use Case 3: Investment Research
```
Scenario: Analyzing property market trends

Input: Run predictions for multiple dates
- Same property details
- Dates: 2015, 2018, 2021, 2024

Analysis: Track price trends over time
Verdict: Identify growth patterns
```

---

## Model Limitations

### What the Model Cannot Do

❌ **Property-Specific Features**
- Cannot account for: number of bedrooms, bathrooms, garden size, parking, condition, renovations

❌ **Recent Market Changes**
- Training data ends 2024
- Cannot predict future market crashes/booms
- Doesn't include very recent sales (last few months)

❌ **Unique Properties**
- Less accurate for: castles, mansions, houseboats, commercial conversions, listed buildings

❌ **Local Context**
- Doesn't know about: new schools, transport links, crime rates, planned developments

### When to Use Professional Valuation

Consider a professional RICS surveyor for:
- Mortgage applications (lender requirement)
- Legal purposes (divorce, probate)
- Insurance valuations
- Commercial properties
- Unique/unusual properties
- Properties over £1 million

---

## Getting Help

### Support Resources

📧 **Technical Issues**
- GitHub Issues: [Report a bug](https://github.com/Franklyn-SWE/UK_HOUSE_PRICE_PREDICTION_V2/issues)

📚 **Documentation**
- [README.md](README.md) - Project overview
- [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) - Future enhancements
- [COMPARISON.md](COMPARISON.md) - Model performance

💬 **Community**
- Discussions: Share use cases and feedback

### Reporting Issues

When reporting problems, include:
1. Error message (if any)
2. Input values used
3. Expected vs actual result
4. Browser and OS version
5. Screenshot (helpful)

---

## Version History

### v1.1 (Current - December 2025)
- ✅ 26% better accuracy (R² 0.247)
- ✅ Added target encoding features
- ✅ Postcode hierarchy extraction
- ✅ Fixed overfitting issues
- ✅ MAPE reduced to 33.8%

### v1.0 (Initial Release)
- Basic CatBoost model
- R² Score: 0.196
- MAPE: 46.1%

---

## Glossary

**CatBoost** - Gradient boosting algorithm specialized for categorical features

**MAPE** - Mean Absolute Percentage Error (average % prediction error)

**R² Score** - Coefficient of determination (% of variance explained, 0-1 scale)

**RMSE** - Root Mean Squared Error (average £ prediction error)

**Target Encoding** - Using average target value (price) for each category

**Postcode Area** - First part of postcode (e.g., "LE17" from "LE17 5AP")

**Freehold** - Ownership of property and land indefinitely

**Leasehold** - Right to occupy property for fixed term (common for flats)

---

## Legal Disclaimer

⚠️ **Important Notice**

This application is for **informational and educational purposes only**. 

**Not Professional Advice:**
- Not a substitute for professional property valuation
- Not suitable for mortgage, legal, or insurance purposes
- Predictions are estimates with significant margin of error (±34%)

**No Warranty:**
- Provided "as is" without guarantees
- Accuracy not guaranteed
- Model limitations acknowledged

**User Responsibility:**
- Verify predictions independently
- Consult professionals for important decisions
- Do not rely solely on these predictions

**Data Sources:**
- Based on historical UK Land Registry data
- Past performance doesn't guarantee future results

For professional valuations, contact:
- RICS Chartered Surveyors
- Estate Agents
- Property Valuation Specialists

---

## Contact & Feedback

**Developer:** Franklyn-SWE

**GitHub:** [@Franklyn-SWE](https://github.com/Franklyn-SWE)

**Project Repository:** [UK_HOUSE_PRICE_PREDICTION_V2](https://github.com/Franklyn-SWE/UK_HOUSE_PRICE_PREDICTION_V2)

**Feedback Welcome:**
- Feature requests
- Bug reports  
- Accuracy feedback
- Use case suggestions

---

**Last Updated:** December 2025 | **Version:** 1.1 | **Model Accuracy:** R² 0.247, MAPE 33.8%

---

*Happy House Hunting! 🏠*
