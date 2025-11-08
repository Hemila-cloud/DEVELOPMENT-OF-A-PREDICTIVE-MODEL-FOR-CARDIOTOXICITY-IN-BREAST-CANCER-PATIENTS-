import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load your preprocessed data
df = pd.read_csv('cardiotoxicity_simple_clean.csv')

def enhanced_feature_engineering(df):
    """Create clinically relevant features and return FULL enhanced dataset"""
    
    # Create a copy to avoid modifying original
    df_enhanced = df.copy()
    
    print("Creating enhanced features...")
    
    # 1. Basic demographics
    df_enhanced['BMI'] = df_enhanced['weight'] / ((df_enhanced['height']/100) ** 2)
    
    # 2. Cardiac structure features
    df_enhanced['LV_size_ratio'] = df_enhanced['LVDd'] / df_enhanced['height']
    df_enhanced['wall_thickness_ratio'] = df_enhanced['PWT'] / df_enhanced['LVDd']
    
    # 3. Treatment burden
    df_enhanced['high_risk_treatment'] = ((df_enhanced['antiHER2'] == 1) | (df_enhanced['AC'] == 1)).astype(int)
    df_enhanced['treatment_burden'] = df_enhanced['antiHER2'] + df_enhanced['AC'] + df_enhanced['RTprev']
    
    # 4. Comorbidity score
    df_enhanced['comorbidity_score'] = df_enhanced['HTA'] + df_enhanced['DL'] + df_enhanced['DM']
    
    # 5. Heart rate categories
    df_enhanced['HR_high'] = (df_enhanced['heart_rate'] > 80).astype(int)
    df_enhanced['HR_very_high'] = (df_enhanced['heart_rate'] > 100).astype(int)
    
    # 6. Time-series features from t1-t1001 columns
    time_cols = [col for col in df_enhanced.columns if col.startswith('t ') and col != 'time']
    
    if len(time_cols) > 0:
        print(f"Processing {len(time_cols)} time-series columns...")
        
        # Use first 12 timepoints for stability
        usable_time_cols = time_cols[:12]  
        time_data = df_enhanced[usable_time_cols]
        
        # Basic trajectory features
        df_enhanced['LVEF_min'] = time_data.min(axis=1)
        df_enhanced['LVEF_max'] = time_data.max(axis=1)
        df_enhanced['LVEF_mean'] = time_data.mean(axis=1)
        df_enhanced['LVEF_std'] = time_data.std(axis=1)
        
        # Critical clinical metrics
        df_enhanced['LVEF_drop_from_baseline'] = df_enhanced['LVEF'] - df_enhanced['LVEF_min']
        df_enhanced['significant_drop'] = (df_enhanced['LVEF_drop_from_baseline'] > 10).astype(int)
        
        # Use t4 as approximate 3-month timepoint
        if len(usable_time_cols) >= 4:
            df_enhanced['LVEF_3month'] = df_enhanced[usable_time_cols[3]]
            df_enhanced['LVEF_change_3mo'] = df_enhanced['LVEF_3month'] - df_enhanced['LVEF']
    
    # Fill any remaining NaN values
    numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
    df_enhanced[numeric_cols] = df_enhanced[numeric_cols].fillna(df_enhanced[numeric_cols].median())
    
    print(f"Enhanced dataset created with {df_enhanced.shape[1]} columns")
    
    return df_enhanced

# Apply enhanced preprocessing
print("Starting enhanced feature engineering...")
df_enhanced = enhanced_feature_engineering(df)

# Display the enhanced dataset info
print("\n" + "="*60)
print("ENHANCED DATASET SUMMARY")
print("="*60)
print(f"Original shape: {df.shape}")
print(f"Enhanced shape: {df_enhanced.shape}")
print(f"New features added: {df_enhanced.shape[1] - df.shape[1]}")

# Show new columns created
original_cols = set(df.columns)
enhanced_cols = set(df_enhanced.columns)
new_columns = list(enhanced_cols - original_cols)

print(f"\nNEW FEATURES CREATED ({len(new_columns)}):")
for i, col in enumerate(new_columns, 1):
    print(f"{i:2}. {col}")

# Display sample of the enhanced data
print(f"\nSAMPLE OF ENHANCED DATA (first 10 patients, first 15 columns):")
print("="*80)
sample_columns = ['CTRCD', 'age', 'BMI', 'heart_rate', 'HR_high', 'LVEF', 'LVEF_min', 
                 'LVEF_drop_from_baseline', 'significant_drop', 'high_risk_treatment', 
                 'treatment_burden', 'comorbidity_score', 'antiHER2', 'AC', 'HTA']
available_sample_cols = [col for col in sample_columns if col in df_enhanced.columns]

print(df_enhanced[available_sample_cols].head(10).round(3))

# Save the FULL enhanced dataset
enhanced_filename = 'cardiotoxicity_dataset_ENHANCED_FULL.csv'
df_enhanced.to_csv(enhanced_filename, index=False)

print(f"\n" + "="*60)
print(f"FULL ENHANCED DATASET SAVED AS: {enhanced_filename}")
print(f"Total columns: {df_enhanced.shape[1]}")
print(f"Total patients: {df_enhanced.shape[0]}")
print("="*60)