# Logistic Regression on Advertising Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Load the dataset
# -----------------------------
ad_data = pd.read_csv("advertising.csv")

print("First 5 rows:")
print(ad_data.head())
print("\n")

# -----------------------------
# 2. Explore the data
# -----------------------------
print("Dataset info:")
print(ad_data.info())
print("\n")

print("Statistical summary:")
print(ad_data.describe())
print("\n")

# -----------------------------
# 3. Data visualization
# -----------------------------

# Histogram of Age
plt.figure(figsize=(8, 5))
ad_data["Age"].hist(bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Jointplot: Age vs Area Income
sns.jointplot(x="Age", y="Area Income", data=ad_data)
plt.show()

# Jointplot: Age vs Daily Time Spent on Site
sns.jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data, kind="kde")
plt.show()

# Pairplot colored by Clicked on Ad
sns.pairplot(
    ad_data[
        [
            "Daily Time Spent on Site",
            "Age",
            "Area Income",
            "Daily Internet Usage",
            "Clicked on Ad",
        ]
    ],
    hue="Clicked on Ad"
)
plt.show()

# -----------------------------
# 4. Prepare the data
# -----------------------------
X = ad_data[[
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage"
]]

y = ad_data["Clicked on Ad"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# -----------------------------
# 5. Train the model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 6. Make predictions
# -----------------------------
predictions = model.predict(X_test)

print("Predictions:")
print(predictions)
print("\n")

# -----------------------------
# 7. Evaluate the model
# -----------------------------
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\n")

print("Classification Report:")
print(classification_report(y_test, predictions))
print("\n")

print("Accuracy Score:")
print(accuracy_score(y_test, predictions))
print("\n")

# -----------------------------
# 8. Simple conclusion
# -----------------------------
print("Conclusion:")
print("We used Logistic Regression to predict whether a user clicked on an ad or not.")
print("The model was trained on selected features and evaluated using classification metrics.")