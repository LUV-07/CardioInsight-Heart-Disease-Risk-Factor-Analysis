from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset directly from UCI (id=45 = Heart Disease dataset)
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

print("Shape:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary stats:")
print(df.describe())

# Save raw data locally so we don't hit the API every run
df.to_csv('data/raw/heart_disease_raw.csv', index=False)
print("\nSaved to data/raw/heart_disease_raw.csv")