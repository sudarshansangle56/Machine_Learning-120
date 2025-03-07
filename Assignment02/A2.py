import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


dataset_path = r'C:\Users\SUDARSHAN\Desktop\ML-LAB-120\Assignment02\traffic_accidents.csv'
df = pd.read_csv(dataset_path)


print(df.head())
print(df.describe())


plt.figure(figsize=(8, 5))
plt.title('Injury Distribution in Traffic Accidents')
sns.histplot(df['injuries_total'], bins=20, kde=True, color='lightcoral')
plt.xlabel('Total Injuries')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(df['crash_hour'], df['injuries_total'], color='lightcoral')
plt.title('Crash Hour vs Total Injuries')
plt.xlabel('Crash Hour')
plt.ylabel('Total Injuries')
plt.show()


label_enc = LabelEncoder()
df['weather_condition'] = label_enc.fit_transform(df['weather_condition'])
df['roadway_surface_cond'] = label_enc.fit_transform(df['roadway_surface_cond'])


X = df[['crash_hour', 'crash_day_of_week', 'crash_month', 'weather_condition', 'roadway_surface_cond']]
y = df[['injuries_total']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_test, color='firebrick')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='dashed')
plt.title('Actual vs Predicted Injuries')
plt.xlabel('Actual Injuries')
plt.ylabel('Predicted Injuries')
plt.show()


print(f'Coefficients: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
