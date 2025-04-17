# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, classification_report
)

# Load dataset
df = pd.read_csv('loan_data.csv')
print(df.info())

# Visualize class distribution for each purpose
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.xticks(rotation=45, ha='right')
plt.title("Loan Purpose vs. Loan Status")
plt.tight_layout()
plt.show()

# Convert categorical to numerical
pre_df = pd.get_dummies(df, columns=['purpose'], drop_first=True)

# Define features and target
X = pre_df.drop('not.fully.paid', axis=1)
y = pre_df['not.fully.paid']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

# Build and train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
labels = ["Fully Paid", "Not fully Paid"]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("Confusion Matrix - Loan Dataset")
plt.show()
