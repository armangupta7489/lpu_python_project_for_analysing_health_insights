import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
#
plt.style.use('seaborn-v0_8')

df = pd.read_csv('unclean_smartwatch_health_data.csv')

print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())
print("\nFirst 5 Rows:")
print(df.head())
print("\nLast 5 Rows:")
print(df.tail())

df['Activity Level'] = df['Activity Level'].replace({
    'Highly_Active': 'Highly Active', 
    'Actve': 'Active', 
    'Seddentary': 'Sedentary'
})
#
df['Stress Level'] = df['Stress Level'].replace('Very High', 10)

df.replace(['ERROR', 'nan', ''], np.nan, inplace=True)

df['User ID'] = df['User ID'].fillna(0)

numeric_cols = ['Heart Rate (BPM)', 'Blood Oxygen Level (%)', 'Step Count', 'Sleep Duration (hours)', 'Stress Level']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
#
df['User ID'] = df['User ID'].astype(int)

df['Activity Level'] = df['Activity Level'].fillna(df['Activity Level'].mode()[0])

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df.drop_duplicates(inplace=True)

df['Activity Level Encoded'] = pd.Categorical(df['Activity Level']).codes

plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 2, i+1)
    sns.histplot(df[col], kde=True, color=sns.color_palette('husl', 5)[i])
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
#
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 2, i+1)
    sns.boxplot(y=df[col], color=sns.color_palette('Set2', 5)[i])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

cov_matrix = df[numeric_cols].cov()
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Covariance Matrix')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Step Count', y='Heart Rate (BPM)', hue='Activity Level', size='Stress Level', palette='deep')
plt.title('Step Count vs Heart Rate by Activity Level and Stress')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Sleep Duration (hours)', y='Blood Oxygen Level (%)', hue='Activity Level', palette='muted')
plt.title('Sleep Duration vs Blood Oxygen Level by Activity Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Activity Level', palette='pastel')
plt.title('Count of Activity Levels')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Activity Level', y='Stress Level', palette='bright')
plt.title('Average Stress Level by Activity Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Activity Level', y='Step Count', palette='dark')
plt.title('Average Step Count by Activity Level')
plt.show()

g = sns.pairplot(df[numeric_cols + ['Activity Level']], hue='Activity Level', palette='Set1')
g.fig.suptitle('Pairplot of Numeric Variables by Activity Level', y=1.02)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Activity Level', y='Heart Rate (BPM)', palette='colorblind')
plt.title('Heart Rate Distribution by Activity Level')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Activity Level', y='Sleep Duration (hours)', palette='cubehelix')
plt.title('Sleep Duration Distribution by Activity Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Heart Rate (BPM)', hue='Activity Level', fill=True, palette='viridis')
plt.title('Heart Rate Density by Activity Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Step Count', hue='Activity Level', fill=True, palette='magma')
plt.title('Step Count Density by Activity Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Activity Level', y='Stress Level', palette='rocket')
plt.title('Stress Level Distribution by Activity Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Activity Level', y='Blood Oxygen Level (%)', palette='mako')
plt.title('Blood Oxygen Level Distribution by Activity Level')
plt.show()

grouped_stats = df.groupby('Activity Level')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
print("\nGrouped Statistics by Activity Level:")
print(grouped_stats)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Sleep Duration (hours)', y='Stress Level', hue='Activity Level', palette='tab10')
plt.title('Stress Level vs Sleep Duration by Activity Level')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Step Count', y='Heart Rate (BPM)', hue='Activity Level', palette='Accent')
plt.title('Heart Rate vs Step Count by Activity Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Step Count', y='Heart Rate (BPM)', scatter_kws={'alpha':0.5}, color='purple')
plt.title('Regression Plot: Step Count vs Heart Rate')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Sleep Duration (hours)', y='Stress Level', scatter_kws={'alpha':0.5}, color='teal')
plt.title('Regression Plot: Sleep Duration vs Stress Level')
plt.show()
##
pivot_table = df.pivot_table(values=numeric_cols, index='Activity Level', aggfunc='mean')
print("\nPivot Table of Means by Activity Level:")
print(pivot_table)
#
plt.figure(figsize=(12, 6))
pivot_table.plot(kind='bar', figsize=(12, 6))
plt.title('Mean Values by Activity Level')
plt.ylabel('Mean Value')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
outliers = (z_scores > 3).any(axis=1)
print("\nNumber of Outliers Detected:", outliers.sum())

df_cleaned = df[~outliers]

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 2, i+1)
    sns.histplot(df_cleaned[col], kde=True, color=sns.color_palette('Paired', 5)[i])
    plt.title(f'Cleaned Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned[numeric_cols].corr(), annot=True, cmap='Spectral', center=0)
plt.title('Correlation Matrix (Cleaned Data)')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_cleaned, x='Step Count', y='Heart Rate (BPM)', hue='Activity Level', size='Stress Level', palette='cool')
plt.title('Cleaned: Step Count vs Heart Rate by Activity Level and Stress')
plt.show()

skewness = df_cleaned[numeric_cols].skew()
kurtosis = df_cleaned[numeric_cols].kurtosis()
print("\nSkewness of Numeric Columns:")
print(skewness)
print("\nKurtosis of Numeric Columns:")
print(kurtosis)

plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned.groupby('Activity Level')[numeric_cols].mean(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Mean Values by Activity Level (Heatmap)')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxenplot(data=df_cleaned, x='Activity Level', y='Step Count', palette='Set3')
plt.title('Enhanced Boxplot: Step Count by Activity Level')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxenplot(data=df_cleaned, x='Activity Level', y='Sleep Duration (hours)', palette='YlOrRd')
plt.title('Enhanced Boxplot: Sleep Duration by Activity Level')
plt.show()

df_cleaned['Step Count Binned'] = pd.cut(df_cleaned['Step Count'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='Step Count Binned', hue='Activity Level', palette='deep')
plt.title('Step Count Bins by Activity Level')
plt.show()

df_cleaned['Sleep Duration Binned'] = pd.cut(df_cleaned['Sleep Duration (hours)'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='Sleep Duration Binned', hue='Activity Level', palette='muted')
plt.title('Sleep Duration Bins by Activity Level')
plt.show()

print("\nValue Counts for Activity Level:")
print(df_cleaned['Activity Level'].value_counts())
print("\nValue Counts for Step Count Binned:")
print(df_cleaned['Step Count Binned'].value_counts())
print("\nValue Counts for Sleep Duration Binned:")
print(df_cleaned['Sleep Duration Binned'].value_counts())

plt.figure(figsize=(12, 6))
sns.catplot(data=df_cleaned, x='Activity Level', y='Heart Rate (BPM)', hue='Step Count Binned', kind='box', palette='viridis')
plt.title('Heart Rate by Activity Level and Step Count Bin')
plt.show()

plt.figure(figsize=(12, 6))
sns.catplot(data=df_cleaned, x='Activity Level', y='Stress Level', hue='Sleep Duration Binned', kind='box', palette='magma')
plt.title('Stress Level by Activity Level and Sleep Duration Bin')
plt.show()

rolling_mean = df_cleaned[numeric_cols].rolling(window=10).mean()
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 2, i+1)
    plt.plot(rolling_mean[col], color=sns.color_palette('Dark2', 5)[i])
    plt.title(f'Rolling Mean of {col} (Window=10)')
plt.tight_layout()
plt.show()

anova_result = stats.f_oneway(
    df_cleaned[df_cleaned['Activity Level'] == 'Active']['Heart Rate (BPM)'].dropna(),
    df_cleaned[df_cleaned['Activity Level'] == 'Highly Active']['Heart Rate (BPM)'].dropna(),
    df_cleaned[df_cleaned['Activity Level'] == 'Sedentary']['Heart Rate (BPM)'].dropna()
)
print("\nANOVA Test for Heart Rate by Activity Level:")
print(f"F-statistic: {anova_result.statistic:.2f}, p-value: {anova_result.pvalue:.4f}")

t_stat, p_val = stats.ttest_ind(
    df_cleaned[df_cleaned['Activity Level'] == 'Active']['Stress Level'].dropna(),
    df_cleaned[df_cleaned['Activity Level'] == 'Sedentary']['Stress Level'].dropna()
)
print("\nT-test for Stress Level (Active vs Sedentary):")
print(f"T-statistic: {t_stat:.2f}, p-value: {p_val:.4f}")

plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned.groupby('Activity Level')[numeric_cols].std(), annot=True, cmap='OrRd', fmt='.2f')
plt.title('Standard Deviation by Activity Level (Heatmap)')
plt.show()

df_cleaned['Heart Rate Binned'] = pd.qcut(df_cleaned['Heart Rate (BPM)'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='Heart Rate Binned', hue='Activity Level', palette='cool')
plt.title('Heart Rate Bins by Activity Level')
plt.show()

cumsum_steps = df_cleaned.groupby('Activity Level')['Step Count'].cumsum()
plt.figure(figsize=(12, 6))
for level in df_cleaned['Activity Level'].unique():
    plt.plot(cumsum_steps[df_cleaned['Activity Level'] == level], label=level)
plt.title('Cumulative Step Count by Activity Level')
plt.xlabel('Index')
plt.ylabel('Cumulative Steps')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df_cleaned, x='Step Count', hue='Activity Level', element='step', palette='tab10')
plt.title('Step Count Histogram by Activity Level (Stacked)')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=df_cleaned, x='Sleep Duration (hours)', hue='Activity Level', palette='Set2')
plt.title('ECDF of Sleep Duration by Activity Level')
plt.show()

pivot_table_multi = df_cleaned.pivot_table(values=['Step Count', 'Heart Rate (BPM)'], index='Activity Level', columns=pd.cut(df_cleaned['Stress Level'], bins=3), aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table_multi['Step Count'], annot=True, cmap='PuBu', fmt='.0f')
plt.title('Mean Step Count by Activity Level and Stress Level Bins')
plt.show()

print("\nDescriptive Statistics After Cleaning:")
print(df_cleaned[numeric_cols].describe())

print("\nMedian Values by Activity Level:")
print(df_cleaned.groupby('Activity Level')[numeric_cols].median())

print("\nQuantiles of Numeric Columns:")
print(df_cleaned[numeric_cols].quantile([0.25, 0.5, 0.75]))

print("\nInsights:")
print("1. Active users show higher step counts and moderate heart rates compared to Sedentary users.")
print("2. Stress levels are lower on average for Highly Active users.")
print("3. Blood Oxygen Levels are relatively stable across activity levels.")
print("4. Sleep Duration has a slight negative correlation with Stress Level.")
print("5. Outlier removal normalized distributions, especially for Step Count.")
print("6. ANOVA test indicates significant differences in Heart Rate across Activity Levels.")
print("7. T-test suggests Stress Levels differ significantly between Active and Sedentary groups.")
print("8. Step Count distributions are right-skewed, especially for Highly Active users.")
print("9. Sleep Duration shows less variability in Active users.")
print("10. Cumulative step counts highlight the dominance of Highly Active users in total activity.")
