import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/heart_disease_raw.csv')

print("=== CLEANING START ===")
print("Initial shape:", df.shape)

#  Convert target 'num' (0-4 severity) into binary: 0 = no disease, 1 = disease present
df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

#  Handle missing values in 'ca' and 'thal'
print(f"\nMissing before imputation -> ca: {df['ca'].isnull().sum()}, thal: {df['thal'].isnull().sum()}")

# ca is a small integer count (0-3) -> impute with mode (most frequent value)
df['ca'] = df['ca'].fillna(df['ca'].mode()[0])

# thal is a coded category -> impute with mode
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

print(f"Missing after imputation -> ca: {df['ca'].isnull().sum()}, thal: {df['thal'].isnull().sum()}")

#  Flag (don't drop) cholesterol == 0 as a data quality issue
zero_chol_count = (df['chol'] == 0).sum()
print(f"\nRows with chol == 0 (flagged, not dropped): {zero_chol_count}")
df['chol_flag'] = np.where(df['chol'] == 0, 'suspect', 'valid')

#  Final check
print("\n=== FINAL SHAPE & MISSING VALUES ===")
print("Shape:", df.shape)
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv('data/processed/heart_disease_cleaned.csv', index=False)
print("\nSaved cleaned data to data/processed/heart_disease_cleaned.csv")
