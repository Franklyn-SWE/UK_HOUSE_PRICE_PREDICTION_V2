#!/bin/bash
# Quick script to train the improved model

echo "======================================================================"
echo "🏠 UK HOUSE PRICE PREDICTION - TRAINING IMPROVED MODEL"
echo "======================================================================"
echo ""
echo "📊 Previous Performance:"
echo "   R² Score:  0.196"
echo "   RMSE:     £327,129"
echo "   MAPE:      46.1%"
echo ""
echo "🎯 Expected Performance:"
echo "   R² Score:  0.40 - 0.55  (2-3x better!)"
echo "   RMSE:     £200k - £250k"
echo "   MAPE:      25% - 35%"
echo ""
echo "======================================================================"
echo ""
echo "Starting training..."
echo ""

cd /workspaces/UK_HOUSE_PRICE_PREDICTION_V2

# Run training
python3 training/train.py

echo ""
echo "======================================================================"
echo "✅ Training Complete!"
echo "======================================================================"
echo ""
echo "📊 View metrics:"
echo "   cat training/metrics.json"
echo ""
echo "🧪 Test the apps:"
echo "   streamlit run app.py"
echo "   python3 gradio_app.py"
echo ""
echo "======================================================================"
