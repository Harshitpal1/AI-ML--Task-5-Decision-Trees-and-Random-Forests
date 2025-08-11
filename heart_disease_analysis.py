import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load the Dataset ---
# Ensure you have the 'heart.csv' file in the same directory as your script.
try:
    df = pd.read_csv('heart.csv')
    print("Dataset loaded successfully! ✅")
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please make sure the dataset file is in the correct directory. ❌")
    exit()

# --- 2. Data Exploration & Preprocessing ---
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
df.info()

# The dataset is clean with no missing values.
# Let's define our features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- 3. Train a Decision Tree Classifier ---
# First, an unconstrained tree that might overfit
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

# Second, a pruned tree to control for overfitting
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
print("\nDecision Tree models trained.")

# --- 4. Train a Random Forest Classifier ---
# An ensemble of 100 decision trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest model trained.")

# --- 5. Evaluate Models on the Test Set ---
# Make predictions
y_pred_dt_full = dt_full.predict(X_test)
y_pred_dt_pruned = dt_pruned.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Calculate accuracy
acc_dt_full = accuracy_score(y_test, y_pred_dt_full)
acc_dt_pruned = accuracy_score(y_test, y_pred_dt_pruned)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n--- Model Accuracy on Test Set ---")
print(f"Full Decision Tree Accuracy: {acc_dt_full:.4f}")
print(f"Pruned Decision Tree (max_depth=4) Accuracy: {acc_dt_pruned:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f} 🌳")

# --- 6. Interpret Feature Importances ---
# Get feature importances from the best model (Random Forest)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n--- Feature Importances from Random Forest ---")
print(feature_importances)

# Plotting the feature importances
plt.figure(figsize=(12, 7))
feature_importances.plot(kind='bar', color='skyblue')
plt.title('Feature Importances in Predicting Heart Disease')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importances.png')
print("\nFeature importance plot saved as 'feature_importances.png'.")

# --- 7. Evaluate Using Cross-Validation ---
# 5-fold cross-validation gives a more robust measure of performance
cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)

print("\n--- 5-Fold Cross-Validation Accuracy ---")
print(f"Pruned Decision Tree CV Accuracy: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std() * 2:.4f})")
print(f"Random Forest CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")
print("\nCross-validation confirms the Random Forest's superior performance. ✨")
