import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs/figures', exist_ok=True)
df = pd.read_csv('data/processed/heart_disease_cleaned.csv')

sns.set_style('whitegrid')

#  Target distribution — how balanced is disease vs no disease?
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Presence Distribution (0=No, 1=Yes)')
plt.savefig('outputs/figures/01_target_distribution.png', bbox_inches='tight')
plt.close()

#  Age distribution by disease presence
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='age', hue='target', kde=True, bins=20)
plt.title('Age Distribution by Heart Disease Presence')
plt.savefig('outputs/figures/02_age_distribution.png', bbox_inches='tight')
plt.close()

#  Sex vs disease presence
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='target', data=df)
plt.title('Disease Presence by Sex (0=Female, 1=Male)')
plt.savefig('outputs/figures/03_sex_vs_disease.png', bbox_inches='tight')
plt.close()

#  Chest pain type vs disease
plt.figure(figsize=(7,5))
sns.countplot(x='cp', hue='target', data=df)
plt.title('Chest Pain Type vs Disease Presence')
plt.savefig('outputs/figures/04_chestpain_vs_disease.png', bbox_inches='tight')
plt.close()

#  Cholesterol vs disease (boxplot) — excluding flagged suspect rows for this view
valid_chol = df[df['chol_flag'] == 'valid']
plt.figure(figsize=(6,4))
sns.boxplot(x='target', y='chol', data=valid_chol)
plt.title('Cholesterol by Disease Presence (suspect rows excluded)')
plt.savefig('outputs/figures/05_chol_vs_disease.png', bbox_inches='tight')
plt.close()

#  Max heart rate vs disease
plt.figure(figsize=(6,4))
sns.boxplot(x='target', y='thalach', data=df)
plt.title('Max Heart Rate Achieved by Disease Presence')
plt.savefig('outputs/figures/06_maxhr_vs_disease.png', bbox_inches='tight')
plt.close()

#  Correlation heatmap (numeric columns only)
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('outputs/figures/07_correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("EDA complete. 7 charts saved to outputs/figures/")
