import pandas as pd
import numpy as np
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import export_graphviz
from scipy.stats import randint

# Load Dataset
bank_data = pd.read_csv("Assignment06/6.csv")

# Data Preprocessing
bank_data['default'] = bank_data['default'].map({'no': 0, 'yes': 1, 'unknown': 0})
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})

# Splitting Data into Features and Target
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualizing Decision Trees
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)

# Hyperparameter Tuning
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(1, 20)
}

rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                 param_distributions=param_dist,
                                 n_iter=5,
                                 cv=5,
                                 random_state=42)
rand_search.fit(X_train, y_train)

# Best Model Selection
best_rf = rand_search.best_estimator_
print('Best hyperparameters:', rand_search.best_params_)

# Evaluate Best Model
y_pred_best = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Additional Metrics
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)

print("Best Model Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Feature Importance
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances.plot.bar()