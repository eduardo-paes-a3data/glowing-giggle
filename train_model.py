import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame with the features and target
df = pd.DataFrame(X, columns=iris.feature_names)
df['class'] = y

# Save the dataset to a CSV file
df.to_csv('dataset.csv', index=False)
print("Dataset saved as 'dataset.csv'")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as 'model.pkl'")
