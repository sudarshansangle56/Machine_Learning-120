import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz  
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
from IPython.display import Image  
import pydotplus
from io import StringIO  
import os

# Load dataset
pima = pd.read_csv("Assignment05/diabetes.csv")

# Print column names to verify
print("Columns in dataset:", pima.columns)

# Define expected columns
expected_columns = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# If column names are incorrect, rename them
if list(pima.columns) != expected_columns:
    print("Renaming columns to match expected format...")
    pima.columns = expected_columns

# Convert all columns to numeric
pima = pima.apply(pd.to_numeric, errors='coerce')

# Define features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Print accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# ---- Decision Tree Visualization ----
dot_data = StringIO()
export_graphviz(
    clf, out_file=dot_data,  
    filled=True, rounded=True,
    special_characters=True, feature_names=feature_cols, class_names=['0', '1']
)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# Save tree visualization
output_image = "diabetes.png"
graph.write_png(output_image)

# Display image if running in Jupyter Notebook
if os.path.exists(output_image):
    display(Image(output_image))
else:
    print(f"Decision Tree image saved as {output_image}, but could not be displayed here.")
