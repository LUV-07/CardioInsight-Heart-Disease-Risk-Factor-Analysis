import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

df = pd.read_csv('data/processed/heart_disease_cleaned.csv')

print("=== CHI-SQUARE TESTS (categorical vs target) ===\n")

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for col in categorical_cols:
    contingency = pd.crosstab(df[col], df['target'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    significant = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"{col:10s} -> chi2={chi2:.2f}, p={p:.4f}  [{significant}]")

print("\n=== T-TESTS (continuous vs target) ===\n")

continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for col in continuous_cols:
    group0 = df[df['target'] == 0][col]
    group1 = df[df['target'] == 1][col]
    t_stat, p_val = ttest_ind(group0, group1)
    significant = "SIGNIFICANT" if p_val < 0.05 else "not significant"
    print(f"{col:10s} -> t={t_stat:.2f}, p={p_val:.4f}  [{significant}]")

print("\n=== SUMMARY ===")
print("Factors with p < 0.05 are statistically associated with heart disease presence.")
print("These are the features to highlight as key findings in your report/dashboard.")