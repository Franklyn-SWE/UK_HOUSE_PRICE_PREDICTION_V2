"""
Quick data quality check and outlier analysis
Run this before training to understand data issues
"""
import pandas as pd
import numpy as np

DATA_PATH = "data/UK_House_Price_Prediction_dataset_2015_to_2024.csv"

def analyze_data_quality():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    
    print(f"\n📊 DATASET SIZE")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\n💰 PRICE STATISTICS")
    print(df['price'].describe())
    print(f"\nPrice range: £{df['price'].min():,.0f} - £{df['price'].max():,.0f}")
    
    # Potential outliers
    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
    print(f"\n⚠️  POTENTIAL OUTLIERS (3*IQR method)")
    print(f"Number of outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Lower bound: £{max(0, lower_bound):,.0f}")
    print(f"Upper bound: £{upper_bound:,.0f}")
    
    # Extreme values
    extreme_low = df[df['price'] < 10000]
    extreme_high = df[df['price'] > 10_000_000]
    print(f"\n🔴 EXTREME VALUES")
    print(f"Prices < £10,000: {len(extreme_low):,} (likely errors)")
    print(f"Prices > £10M: {len(extreme_high):,}")
    
    if len(extreme_low) > 0:
        print(f"  → Examples: £{extreme_low['price'].min():,.0f}, £{extreme_low['price'].median():,.0f}")
    
    print(f"\n📍 CATEGORICAL CARDINALITY")
    for col in ['postcode', 'property_type', 'street', 'locality', 'town', 'district', 'county']:
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        print(f"{col:15s}: {n_unique:7,} unique values | {n_missing:6,} missing ({n_missing/len(df)*100:5.2f}%)")
    
    print(f"\n🏠 PROPERTY TYPE DISTRIBUTION")
    prop_counts = df['property_type'].value_counts()
    for prop_type, count in prop_counts.items():
        pct = count / len(df) * 100
        print(f"  {prop_type}: {count:7,} ({pct:5.2f}%)")
    
    # Unknown/invalid property types
    valid_types = ['D', 'S', 'T', 'F', 'O']
    invalid = df[~df['property_type'].isin(valid_types)]
    if len(invalid) > 0:
        print(f"\n  ⚠️  Invalid property types: {len(invalid):,}")
        print(f"     Types: {invalid['property_type'].unique()}")
    
    print(f"\n🆕 NEW BUILD vs RESALE")
    print(df['new_build'].value_counts())
    
    print(f"\n🏛️  FREEHOLD vs LEASEHOLD")
    print(df['freehold'].value_counts())
    
    print(f"\n❌ MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col:15s}: {count:7,} ({count/len(df)*100:5.2f}%)")
    else:
        print("  No missing values ✅")
    
    print(f"\n🔁 DUPLICATE ROWS")
    duplicates = df.duplicated().sum()
    print(f"Exact duplicates: {duplicates:,}")
    
    # Potential duplicate sales (same property, same date)
    dup_sales = df.duplicated(subset=['postcode', 'street', 'date']).sum()
    print(f"Duplicate sales (same property+date): {dup_sales:,}")
    
    print(f"\n📅 TEMPORAL DISTRIBUTION")
    yearly = df.groupby(df['date'].dt.year).size()
    for year, count in yearly.items():
        print(f"  {year}: {count:7,} sales")
    
    print(f"\n💡 RECOMMENDATIONS")
    recommendations = []
    
    if len(extreme_low) > 0:
        recommendations.append(f"✓ Remove {len(extreme_low):,} properties < £10k (likely errors)")
    
    if len(extreme_high) > 50:
        recommendations.append(f"✓ Review {len(extreme_high):,} properties > £10M (may be commercial)")
    
    if len(outliers) > len(df) * 0.05:
        recommendations.append(f"✓ Consider outlier treatment for {len(outliers):,} extreme values")
    
    if dup_sales > 0:
        recommendations.append(f"✓ Remove {dup_sales:,} duplicate sales")
    
    if len(invalid) > 0:
        recommendations.append(f"✓ Clean {len(invalid):,} invalid property types")
    
    high_cardinality = ['postcode', 'street', 'locality']
    recommendations.append(f"✓ Use target encoding for high-cardinality features: {', '.join(high_cardinality)}")
    
    recommendations.append("✓ Add postcode hierarchy features (area, district)")
    recommendations.append("✓ Add temporal features (year trend, seasonality)")
    
    if len(recommendations) > 0:
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "=" * 70)
    print(f"🎯 ESTIMATED MODEL PERFORMANCE")
    print("=" * 70)
    print(f"Current R² (baseline):  ~0.20")
    print(f"With improvements:      ~0.40-0.55")
    print(f"Production target:       >0.50")
    print("=" * 70)


if __name__ == "__main__":
    analyze_data_quality()
