import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

dataset_name = r'C:\Users\SUDARSHAN\Desktop\ML-LAB-120\traffic_accidents.csv'

df = pd.read_csv(dataset_name)

print("Dataset Head:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

for column in df.select_dtypes(include=['float64', 'int64']):
    df[column] = df[column].fillna(df[column].median())

for column in df.select_dtypes(include=['object']):
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nSummary Statistics:")
print(df.describe())

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

sns.pairplot(df, diag_kind='kde', markers='+')
plt.suptitle("Pairplot of Dataset", y=1.02)
plt.show()

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

for column in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column], color='green')
    plt.title(f'Boxplot of {column}')
    plt.show()

if len(numerical_columns) >= 2:
    t_stat, p_value = stats.ttest_ind(df[numerical_columns[0]], df[numerical_columns[1]])
    print(f"\nT-Test Results between {numerical_columns[0]} and {numerical_columns[1]}:")
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

cleaned_dataset_name = r'C:\Users\SUDARSHAN\Desktop\ML-LAB-120\cleaned_traffic_accidents.csv'
df.to_csv(cleaned_dataset_name, index=False)
print(f"\nCleaned dataset saved as {cleaned_dataset_name}")
