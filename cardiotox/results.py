import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Define the UltimateCardiotoxicityPredictor class as it was in main.py to use its methods
class UltimateCardiotoxicityPredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.final_auc = 0
        self.research_baseline_auc = 0.81  # CHECK-HEART paper AUC

    def clean_data(self, X):
        """Remove infinite values and handle missing data properly"""
        # print("Cleaning data...") # Suppress print for cleaner output
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        inf_count = (X == np.inf).sum().sum() + (X == -np.inf).sum().sum()
        # if inf_count > 0:
            # print(f"Removed {inf_count} infinite values")

        X_clean = X_clean.fillna(X_clean.median())
        if X_clean.isnull().any().any():
            # print(f"Warning: {X_clean.isnull().sum().sum()} NaN values remaining")
            X_clean = X_clean.fillna(0)
        return X_clean

    def create_super_features(self, df):
        """Create features that NO research paper has used together"""
        # print("Creating SUPER FEATURES beyond current research...") # Suppress print

        # 1. Treatment response
        if 'LVEF_change_3mo' in df.columns:
            df['early_responder'] = (df['LVEF_change_3mo'] < -5).astype(int)
            df['stable_responder'] = (abs(df['LVEF_change_3mo']) <= 5).astype(int)

        # 2. Remodeling index
        if all(col in df.columns for col in ['LVDd', 'PWT', 'LVEF']):
            df['remodeling_index'] = (df['LVDd'] * df['PWT']) / df['LVEF'].replace(0, 1)

        # 3. Diastolic-systolic mismatch
        if all(col in df.columns for col in ['mitral_E_A_ratio', 'LVEF', 'E_e_ratio']):
            df['DS_mismatch'] = ((df['E_e_ratio'] > 10) & (df['LVEF'] > 55)).astype(int)
            df['DS_mismatch_score'] = df['E_e_ratio'] / df['LVEF'].replace(0, 1)

        # 4. Toxicity burden
        df['toxicity_burden'] = (
            df.get('antiHER2', 0) * 2 +
            df.get('AC', 0) * 1.5 +
            df.get('RTprev', 0) * 1 +
            df.get('HTA', 0) * 0.5 +
            df.get('DM', 0) * 0.5
        )

        # 5. Time-series instability
        time_cols = [col for col in df.columns if col.startswith('t ')]
        if len(time_cols) > 3:
            time_data = df[time_cols[:6]].copy()
            time_data = time_data.replace([np.inf, -np.inf], np.nan).fillna(time_data.median())
            means = time_data.mean(axis=1).replace(0, 1)
            df['LVEF_instability'] = time_data.std(axis=1) / means
            df['early_trend'] = (time_data.iloc[:, 3] - time_data.iloc[:, 0]) / 3

        return df

    def select_research_beating_features(self, df):
        research_features = ['age', 'heart_rate', 'LVEF', 'antiHER2', 'AC', 'HTA', 'DM']
        enhanced_features = ['BMI', 'LVEF_min', 'LVEF_drop_from_baseline', 'LVEF_change_3mo',
                             'high_risk_treatment', 'treatment_burden', 'comorbidity_score']
        novel_features = ['early_responder', 'remodeling_index', 'DS_mismatch', 'DS_mismatch_score',
                          'toxicity_burden', 'LVEF_instability', 'early_trend']
        
        all_potential = research_features + enhanced_features + novel_features
        available_features = [f for f in all_potential if f in df.columns]

        return available_features


# Instantiate the predictor to use its methods for preprocessing
predictor = UltimateCardiotoxicityPredictor()

# Load the dataset
df = pd.read_csv('cardiotoxicity_dataset_ENHANCED_FULL.csv')

# Apply the same feature engineering as in main.py
df_super = predictor.create_super_features(df)
features = predictor.select_research_beating_features(df_super)
X = df_super[features]
y = df_super['CTRCD']

# Ensure target is binary
if len(y.unique()) > 2:
    y = (y > 0).astype(int)

# Clean the data
X_clean = predictor.clean_data(X)

# Split the data using the same parameters as in main.py
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

# Load the trained model
loaded_model_data = joblib.load('CHAMPION_cardiotoxicity_model.pkl')
model = loaded_model_data['model']

# Make predictions on the test set
y_pred_proba = model.predict_proba(predictor.clean_data(X_test))[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int) # Assuming 0.5 as the default threshold

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("Confusion matrix plot generated successfully.")