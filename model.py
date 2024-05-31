import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

dataset_path = r'E:\breast cancer detection\data.csv'

data = pd.read_csv(dataset_path)

# # Displaying the head of the table
# print(data.head())

# # Display basic information about the dataset
# print(data.info())

# # Check for missing values
# print(data.isnull().sum())

# # Display summary statistics
# print(data.describe())

# # Check for duplicates
# print(data.duplicated().sum())

# Removing duplicates if any
data = data.drop_duplicates()

# Encode the target variable
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Select only the features you will use in the frontend form
selected_features = ['radius_mean', 'texture_mean', 'radius_worst', 'texture_worst', 'symmetry_mean', 'symmetry_worst']
X = data[selected_features]
y = data['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1-Score: {f1}')
# print(f'ROC AUC: {roc_auc}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix:\n', cm)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_model.predict(X_test_scaled)
y_pred_best_proba = best_model.predict_proba(X_test_scaled)[:, 1]

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)
best_roc_auc = roc_auc_score(y_test, y_pred_best_proba)

# print(f'Best Accuracy: {best_accuracy}')
# print(f'Best Precision: {best_precision}')
# print(f'Best Recall: {best_recall}')
# print(f'Best F1-Score: {best_f1}')
# print(f'Best ROC AUC: {best_roc_auc}')

# Save the model and scaler
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model training completed and saved.")
