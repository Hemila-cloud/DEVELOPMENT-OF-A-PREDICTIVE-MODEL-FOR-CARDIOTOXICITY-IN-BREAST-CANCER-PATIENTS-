import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('final_cardiotoxicity_dataset_COMPLETE.csv')

# 1. Remove completely empty columns
empty_cols = [col for col in df.columns if df[col].isna().all()]
df = df.drop(columns=empty_cols)

# 2. Remove Unnamed columns
unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
df = df.drop(columns=unnamed_cols)

# 3. Fill missing values with 0 for binary, median for numeric
binary_cols = ['CTRCD', 'antiHER2', 'HTA', 'DL', 'DM', 'smoker', 'exsmoker']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Fill other numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 4. Prepare for modeling
X = df.drop('CTRCD', axis=1)
y = df['CTRCD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save
df.to_csv('cardiotoxicity_simple_clean.csv', index=False)
X_train.to_csv('X_train_simple.csv', index=False)
X_test.to_csv('X_test_simple.csv', index=False)
y_train.to_csv('y_train_simple.csv', index=False)  
y_test.to_csv('y_test_simple.csv', index=False)

print("Simple preprocessing complete!")
print(f"Final shape: {df.shape}")
print(f"Cardiotoxicity rate: {y.mean():.2%}")