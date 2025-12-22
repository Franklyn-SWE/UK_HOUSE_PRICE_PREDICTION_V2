# UK House Price Prediction v1.1 🏠📈

A machine learning application that predicts UK house prices using CatBoost regression with advanced feature engineering and target encoding.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.1-orange.svg)](https://github.com/Franklyn-SWE/UK_HOUSE_PRICE_PREDICTION_V2/releases/tag/v1.1)

🌐 **[Live Demo →](https://ukhousepricepredictionv2.streamlit.app/)**

## 🎯 Performance Metrics (v1.1)

| Metric | v1.1 (Current) | v1.0 (Baseline) | Improvement |
|--------|----------------|-----------------|-------------|
| **R² Score** | 0.247 | 0.196 | +26% |
| **RMSE** | £316,629 | £327,129 | -3.2% |
| **MAE** | £101,505 | £105,441 | -3.7% |
| **Median AE** | £53,727 | £55,848 | -3.8% |
| **MAPE** | 33.8% | 46.1% | -27% |

**Baseline Improvement:** 37.5% better than median baseline prediction

## ✨ Key Features

### v1.1 Improvements (Latest)
- ✅ **Target Encoding**: Mean price encoding for town, district, county, and property type
- ✅ **Postcode Hierarchy**: Extract postcode area and district for better geographic representation
- ✅ **Enhanced Time Features**: Added `years_since_2015` to capture market trends
- ✅ **Optimized Hyperparameters**: Fine-tuned depth, learning rate, and regularization
- ✅ **Bug Fixes**: Fixed scale mismatch in target encoding and removed overfitting features
- ✅ **Balanced Feature Importance**: No single feature dominates (top feature: 23%)

### Core Features
- 🤖 **CatBoost Regression**: Advanced gradient boosting with categorical feature support
- 📊 **90,000+ Training Records**: UK house sales data from 2015-2024
- 🎨 **Interactive UI**: Streamlit and Gradio interfaces
- 🔄 **Automatic Training**: Model trains on deployment if not cached
- 📈 **Comprehensive Metrics**: R², RMSE, MAE, MAPE, and more

![UK House Price Prediction App Interface](https://github.com/Franklyn-SWE/UK_HOUSE_PRICE_PREDICTION_APP/blob/main/images/uk_house_pred_app_2.png)  
*The interface of the UK House Price Prediction application, displaying input features and user-friendly design.*

![Simulation of UK House Price Prediction App](https://github.com/Franklyn-SWE/UK_HOUSE_PRICE_PREDICTION_APP/blob/main/images/uk_house_pred_ui.png)  
*A simulation showcasing the app's prediction process and output results.*

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Franklyn-SWE/UK_HOUSE_PRICE_PREDICTION_V2.git
cd UK_HOUSE_PRICE_PREDICTION_V2

# Install dependencies
pip install -r requirements.txt

# Train the model (takes ~5-6 minutes)
python training/train.py

# Run Streamlit app
streamlit run app.py

# Or run Gradio app
python gradio_app.py
```

## 📊 Model Architecture

### Feature Engineering Pipeline
```
Raw Data
  ↓
Date Features (year, month, quarter, dayofweek, month_end, years_since_2015)
  ↓
Postcode Hierarchy (postcode_area, postcode_district)
  ↓
Target Encoding (town_mean_price, district_mean_price, county_mean_price, property_type_mean_price)
  ↓
CatBoost Model (depth=8, lr=0.03, iterations=2000, l2_reg=10)
  ↓
Log-Transformed Predictions → Exponentiated to Original Scale
```

### Top 10 Features by Importance
1. **property_type** (22.93%) - Property classification (D/S/T/F/O)
2. **town_mean_price** (12.83%) - Average price in town
3. **property_type_mean_price** (9.93%) - Average price by type
4. **locality** (6.68%) - Local area name
5. **postcode_area** (6.40%) - Postcode area code
6. **district_mean_price** (6.39%) - Average price in district
7. **county** (4.66%) - County location
8. **town** (4.62%) - Town name
9. **district** (4.45%) - District name
10. **postcode_district** (4.24%) - Postcode district code

## 📁 Project Structure

```
UK_HOUSE_PRICE_PREDICTION_V2/
├── app.py                      # Streamlit web application
├── gradio_app.py              # Gradio alternative interface
├── requirements.txt           # Python dependencies
├── postBuild                  # Deployment build script
├── .python-version            # Python version specification
├── data/
│   └── UK_House_Price_Prediction_dataset_2015_to_2024.csv
├── training/
│   ├── train.py              # Main training script (v1.1)
│   ├── train_improved.py     # Standalone improved version
│   ├── analyze_data.py       # Data quality analysis tool
│   └── metrics.json          # Model performance metrics
├── docs/
│   ├── IMPROVEMENT_GUIDE.md           # Roadmap for future improvements
│   ├── IMPLEMENTATION_SUMMARY.md      # Implementation details
│   ├── COMPARISON.md                  # v1.0 vs v1.1 comparison
│   └── BUG_FIX_SCALE_MISMATCH.md     # Scale bug documentation
└── images/                    # UI screenshots
```

## 🔧 Technical Details

### Model Hyperparameters
```python
CatBoostRegressor(
    loss_function="RMSE",
    iterations=2000,           # Increased for better convergence
    depth=8,                   # Balanced complexity
    learning_rate=0.03,        # Conservative for stability
    l2_leaf_reg=10,           # L2 regularization
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=100,
    verbose=100
)
```

### Data Split
- **Training:** 2015-2022 (81,179 records)
- **Validation:** 2023 (6,832 records)
- **Test:** 2024 (1,633 records)

### Target Transformation
- Applied `log1p` transformation to handle price skewness
- Predictions exponentiated back to original scale

## 📈 Model Performance Analysis

### What Works Well
- ✅ Property type classification (22.93% importance)
- ✅ Geographic encoding (town/district/county averages)
- ✅ Postcode hierarchy features
- ✅ Balanced feature distribution (no overfitting)

### Known Limitations
- ⚠️ Limited property characteristic data (no bedroom/bathroom counts)
- ⚠️ No economic indicators (interest rates, inflation)
- ⚠️ No geospatial features (distance to amenities)
- ⚠️ MAPE still high at 33.8% (target: <25%)

### Future Improvements (Roadmap to R² > 0.50)
See [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) for detailed recommendations:
- Add property size features (+0.10-0.15 R²)
- Implement ensemble models (+0.03-0.06 R²)
- Add geospatial features (+0.05-0.10 R²)
- Integrate economic indicators (+0.02-0.04 R²)

## 🌐 Deployment

### Live Demo
Access the deployed application at: [Your Streamlit App URL]

### Deploy Your Own

**Streamlit Community Cloud:**
1. Fork this repository
2. Go to https://share.streamlit.io/
3. Connect your GitHub account
4. Select this repo and `app.py`
5. Click "Deploy" (takes ~8-10 minutes first time)

**Render/Heroku:**
- The `postBuild` script automatically trains the model on deployment
- Training takes 5-6 minutes but is cached for subsequent restarts

## 🐛 Debugging History

### v1.1 Bug Fixes
1. **Scale Mismatch Bug** (Fixed)
   - Issue: Target encodings in raw price scale vs log-transformed target
   - Solution: Applied `np.log1p()` to all target-encoded features
   - Impact: Fixed extreme feature importance (75% → balanced)

2. **Overfitting Bug** (Fixed)
   - Issue: `postcode_mean_price` had 75% feature importance
   - Solution: Removed postcode-level encoding, kept town/district/county
   - Impact: Model generalizes better, R² improved from 0.03 → 0.247

See [BUG_FIX_SCALE_MISMATCH.md](BUG_FIX_SCALE_MISMATCH.md) for details.

## 📚 Documentation

- [USER_MANUAL.md](USER_MANUAL.md) - Complete user guide for the app
- [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) - 8-step improvement roadmap
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Complete implementation details
- [COMPARISON.md](COMPARISON.md) - Before/after performance comparison
- [BUG_FIX_SCALE_MISMATCH.md](BUG_FIX_SCALE_MISMATCH.md) - Scale bug fix documentation

## 🌐 Deployment

### Live Application
**🚀 Try it now:** [https://ukhousepricepredictionv2.streamlit.app/](https://ukhousepricepredictionv2.streamlit.app/)

Features:
- ✅ No installation required
- ✅ Instant predictions
- ✅ Mobile-friendly interface
- ✅ Always up-to-date with latest model

### Deploy Your Own
- [BUG_FIX_SCALE_MISMATCH.md](BUG_FIX_SCALE_MISMATCH.md) - Scale bug fix documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: UK Land Registry House Price Data (2015-2024)
- Framework: CatBoost by Yandex
- UI: Streamlit and Gradio
- Deployment: Streamlit Community Cloud

## 📞 Contact

**Franklyn-SWE**
- GitHub: [@Franklyn-SWE](https://github.com/Franklyn-SWE)
- Project: [UK_HOUSE_PRICE_PREDICTION_V2](https://github.com/Franklyn-SWE/UK_HOUSE_PRICE_PREDICTION_V2)

---

**Version 1.1** | Released December 2025 | 26% Better Performance 🚀
