import pandas as pd

df = pd.read_csv('data/processed/heart_disease_cleaned.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

print("=== SEGMENTATION ANALYSIS ===\n")

# 1. Age bracket segmentation
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 100], labels=['<40', '40-55', '55+'])

print("--- Disease rate by age group ---")
age_seg = df.groupby('age_group')['target'].agg(['mean', 'count'])
age_seg['disease_rate_%'] = (age_seg['mean'] * 100).round(1)
print(age_seg[['disease_rate_%', 'count']])

# 2. Age group x Sex
print("\n--- Disease rate by age group and sex (0=Female, 1=Male) ---")
age_sex_seg = df.groupby(['age_group', 'sex'])['target'].agg(['mean', 'count'])
age_sex_seg['disease_rate_%'] = (age_sex_seg['mean'] * 100).round(1)
print(age_sex_seg[['disease_rate_%', 'count']])

# 3. Chest pain type disease rate (your strongest categorical finding)
print("\n--- Disease rate by chest pain type ---")
cp_seg = df.groupby('cp')['target'].agg(['mean', 'count'])
cp_seg['disease_rate_%'] = (cp_seg['mean'] * 100).round(1)
print(cp_seg[['disease_rate_%', 'count']])

# 4. Thal (thalassemia result) disease rate
print("\n--- Disease rate by thal result ---")
thal_seg = df.groupby('thal')['target'].agg(['mean', 'count'])
thal_seg['disease_rate_%'] = (thal_seg['mean'] * 100).round(1)
print(thal_seg[['disease_rate_%', 'count']])

# 5. Number of major vessels (ca) disease rate
print("\n--- Disease rate by number of major vessels (ca) ---")
ca_seg = df.groupby('ca')['target'].agg(['mean', 'count'])
ca_seg['disease_rate_%'] = (ca_seg['mean'] * 100).round(1)
print(ca_seg[['disease_rate_%', 'count']])

# 6. Highest-risk combined segment: older males with asymptomatic chest pain (cp=4)
print("\n--- Highest-risk segment check: age 55+, male, cp=4 ---")
high_risk = df[(df['age_group'] == '55+') & (df['sex'] == 1) & (df['cp'] == 4)]
if len(high_risk) > 0:
    rate = high_risk['target'].mean() * 100
    print(f"Segment size: {len(high_risk)}, Disease rate: {rate:.1f}%")
else:
    print("No rows match this exact segment.")

# 7. Compare to overall baseline disease rate
overall_rate = df['target'].mean() * 100
print(f"\nOverall baseline disease rate (all patients): {overall_rate:.1f}%")

# Save segmentation summary for use in dashboard/README
df.to_csv('data/processed/heart_disease_with_segments.csv', index=False)
print("\nSaved segmented dataset to data/processed/heart_disease_with_segments.csv")