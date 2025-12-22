# 🔧 Critical Bug Fix - Scale Mismatch in Target Encoding

## 🐛 **Problem Identified**

The first training run failed (R² = 0.028, worse than baseline 0.196) due to a **scale mismatch** between features and target.

### **Root Cause:**

```python
# WRONG: Target encoding in raw price scale
postcode_mean_price = £240,000  # Raw scale

# Target variable in log scale
y_train = np.log1p(train_df["price"])  # log(240,000) ≈ 12.4
```

**The model tried to learn from:**
- Features with values ~12 (log scale: years, months)
- Features with values ~240,000 (raw price scale: encoded means)
- Target with values ~12 (log scale)

**Result:** The huge scale difference (1 vs 240,000) caused the model to:
1. Overweight the encoded features (68% importance!)
2. Learn incorrect patterns
3. Perform worse than baseline

---

## ✅ **Solution Applied**

Transform target-encoded features to **match the log scale** of the target:

```python
# CORRECT: Target encoding in log scale
col_means["smoothed_mean_log"] = np.log1p(col_means["smoothed_mean"])

# Now everything is in the same scale:
postcode_mean_price = 12.4  # Log scale
y_train = 12.4              # Log scale
```

---

## 🔧 **Changes Made**

### **1. training/train.py**

#### Added log transformation to target encoding:
```python
# Apply log1p transformation to match target scale
col_means["smoothed_mean_log"] = np.log1p(col_means["smoothed_mean"])
encoding_map[col] = dict(zip(col_means[col], col_means["smoothed_mean_log"]))
```

#### Updated fallback values:
```python
global_mean_log = np.log1p(train_df["price"].mean())
df_set["postcode_mean_price"] = df_set["postcode"].map(...).fillna(global_mean_log)
```

#### Adjusted hyperparameters to prevent overfitting:
```python
depth=8,              # Back to 8 (strong features don't need deep trees)
l2_leaf_reg=10,       # Increased regularization (from 5)
bagging_temperature=1.0,  # Default (less aggressive)
random_strength=1.0,      # Default (less aggressive)
```

### **2. app.py & gradio_app.py**

Updated inference to use log-transformed encoding:
```python
train_mean_log = np.log1p(train_mean)
df["postcode_mean_price"] = df["postcode"].map(...).fillna(train_mean_log)
```

---

## 📊 **Expected Impact**

### Before Fix:
- R² = 0.028 (broken)
- RMSE = £360k
- MAPE = 56.7%

### After Fix (Expected):
- R² = **0.40 - 0.55** (2-3x better than baseline 0.196)
- RMSE = **£200k - £250k** (much better)
- MAPE = **25% - 35%** (acceptable)

---

## 🎯 **Why This Matters**

### Scale Consistency in Machine Learning

When features and targets have different scales, the model:
1. **Learns biased weights** - Large-scale features dominate
2. **Converges slowly** - Gradient descent struggles
3. **Overfits easily** - Focus on wrong patterns

### Best Practice:
✅ **All features should be on similar scales**
- Numeric features: Normalize or standardize
- Categorical encodings: Match target scale
- Target: Log-transform for prices (reduces skewness)

---

## 🚀 **Next Steps**

1. **Retrain the model:**
   ```bash
   python training/train.py
   ```

2. **Verify improvements:**
   - R² > 0.40 (minimum)
   - MAPE < 35%
   - Feature importance still shows new features at top

3. **Check feature importance:**
   - `postcode_mean_price` should still be #1
   - But with balanced influence (not 68%!)

---

## 📈 **Feature Importance (Expected After Fix)**

### Before Fix:
```
postcode_mean_price:        68% ⚠️ TOO HIGH - overfitting!
property_type:              16%
property_type_mean_price:    4%
```

### After Fix (Expected):
```
postcode_mean_price:        25-35% ✅ Balanced
postcode:                   15-20% ✅
property_type:              10-15% ✅
town_mean_price:             8-12% ✅
property_type_mean_price:    5-8%  ✅
```

**Total from new features:** 40-50% (healthy, not dominating)

---

## 🎓 **Key Learnings**

1. **Always check feature scales** - Especially with derived features
2. **Target encoding must match target scale** - Log → log, raw → raw
3. **High feature importance ≠ good** - 68% on one feature = overfitting
4. **Test on validation set** - Would have caught this earlier

---

## ✅ **Verification Checklist**

After retraining, verify:
- [ ] R² > 0.40 (2x better than baseline 0.196)
- [ ] MAPE < 35%
- [ ] Feature importance is balanced (no single feature >40%)
- [ ] Validation score close to training score (no overfitting)
- [ ] Predictions make sense on sample properties

---

**Status:** ✅ Bug fixed, ready for retraining
**Expected improvement:** 2-3x better performance
**File:** Run `python training/train.py` to test
