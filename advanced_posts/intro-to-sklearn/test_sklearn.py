import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('default')
sns.set_theme()

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Create features
X = np.random.normal(size=(n_samples, 2))
# Create two clusters
y = (X[:, 0]**2 + X[:, 1]**2 > 2).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Look at the distribution of scaled features
plt.figure(figsize=(10, 5))
sns.histplot(X_train_scaled[:, 0], bins=30, kde=True)
plt.title('Distribution of Scaled Feature 1')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Print the results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create a mesh grid to visualize the decision boundary
def plot_decision_boundary(X, y, model, scaler):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Scale the mesh points
    mesh_points = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    
    # Make predictions
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('Random Forest Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    return plt.gcf()

# Plot the decision boundary
plot_decision_boundary(X_test, y_test, rf_model, scaler)
plt.show()
