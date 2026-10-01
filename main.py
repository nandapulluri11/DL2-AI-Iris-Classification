# 1. Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 2. Load the Iris dataset
# ============================================

iris = load_iris()

print("Dataset loaded successfully!")


# ============================================
# 3. Separate features and target
# ============================================

X = iris.data
y = iris.target


# ============================================
# 4. Understand the dataset
# ============================================

print("\n--- Dataset Information ---")

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("Feature names:")
for feature in iris.feature_names:
    print("-", feature)

print("Target names:", iris.target_names)


# ============================================
# 5. Split data into training and testing
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- Data Split ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 6. Feature Scaling
# ============================================

scaler = StandardScaler()

# Learn scaling from training data
# and transform the training data
X_train = scaler.fit_transform(X_train)

# Use the same scaling rules
# on the testing data
X_test = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ============================================
# 7. Create the KNN model
# ============================================

model = KNeighborsClassifier(n_neighbors=5)

print("\nKNN model created!")
print("Number of neighbors (K): 5")


# ============================================
# 8. Train the model
# ============================================

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================
# 9. Make predictions
# ============================================

predictions = model.predict(X_test)

print("Predictions completed!")


# ============================================
# 10. Calculate Accuracy
# ============================================

accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Accuracy ---")
print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")


# ============================================
# 11. Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, predictions)

print("\n--- Confusion Matrix ---")
print(cm)


# ============================================
# 12. Classification Report
# ============================================

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)


# ============================================
# 13. Final Result
# ============================================

print("--- Project Completed Successfully ---")