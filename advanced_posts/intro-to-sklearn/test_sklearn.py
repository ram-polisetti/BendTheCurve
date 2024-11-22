"""
Introduction to Machine Learning with Scikit-learn

This tutorial demonstrates fundamental machine learning concepts using scikit-learn:
1. Data Generation and Preprocessing
2. Model Training and Evaluation
3. Visualization of Results

We'll create a simple classification problem and solve it using Random Forest.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent visual styling for all plots
plt.style.use('default')
sns.set_theme()

# Step 1: Generate Sample Data
# --------------------------
# Create a synthetic classification dataset where points inside and 
# outside a circle are labeled differently
np.random.seed(42)  # For reproducibility
n_samples = 1000

# Generate random points in 2D space
X = np.random.normal(size=(n_samples, 2))
# Label points: 1 if outside the circle (radius = sqrt(2)), 0 if inside
y = (X[:, 0]**2 + X[:, 1]**2 > 2).astype(int)

# Step 2: Data Splitting and Preprocessing
# -------------------------------------
# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features to have zero mean and unit variance
# This is important for many ML algorithms
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Visualize the distribution of scaled features
plt.figure(figsize=(12, 5))

# Plot histogram of first feature
plt.subplot(1, 2, 1)
sns.histplot(X_train_scaled[:, 0], bins=30, kde=True)
plt.title('Distribution of Scaled Feature 1')
plt.xlabel('Value')
plt.ylabel('Count')

# Plot histogram of second feature
plt.subplot(1, 2, 2)
sns.histplot(X_train_scaled[:, 1], bins=30, kde=True)
plt.title('Distribution of Scaled Feature 2')
plt.xlabel('Value')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Step 3: Model Training
# -------------------
# Initialize and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    random_state=42    # For reproducibility
)
rf_model.fit(X_train_scaled, y_train)

# Step 4: Model Evaluation
# ---------------------
# Make predictions on test set
y_pred = rf_model.predict(X_test_scaled)

# Print detailed classification metrics
print("\nModel Performance Metrics:")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Visualize Decision Boundary
# -------------------------------
def plot_decision_boundary(X, y, model, scaler):
    """
    Visualize the model's decision boundary along with the data points.
    
    Args:
        X: Feature matrix
        y: Target labels
        model: Trained classifier
        scaler: Fitted StandardScaler
    """
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Scale the mesh points using the same scaler used for training
    mesh_points = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    
    # Get predictions for all mesh points
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, 
                         cmap='RdYlBu', alpha=0.8)
    plt.colorbar(scatter)
    plt.title('Random Forest Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return plt.gcf()

# Plot the decision boundary with test data
plot_decision_boundary(X_test, y_test, rf_model, scaler)
plt.show()

# Optional: Feature Importance Analysis
# --------------------------------
feature_importance = pd.DataFrame({
    'feature': [f'Feature {i+1}' for i in range(2)],
    'importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print("=" * 50)
print(feature_importance.sort_values('importance', ascending=False))
