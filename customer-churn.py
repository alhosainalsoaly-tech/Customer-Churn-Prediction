# Customer Churn Prediction
# A simple Data Science and Machine Learning project

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# 1. Create the dataset
# -----------------------------

# I made a small dataset with some customer information.
# Churn = 1 means the customer left.
# Churn = 0 means the customer stayed.

data = {
    "Age": [
        22, 35, 28, 45, 31,
        50, 26, 40, 29, 55,
        24, 38, 33, 48, 27,
        52, 30, 42, 25, 46
    ],

    "Monthly_Charges": [
        30, 70, 45, 90, 55,
        100, 35, 85, 50, 110,
        40, 75, 65, 95, 38,
        105, 60, 80, 42, 88
    ],

    "Tenure": [
        2, 24, 8, 36, 12,
        48, 5, 30, 10, 60,
        4, 28, 18, 42, 6,
        54, 15, 32, 7, 40
    ],

    "Churn": [
        1, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0
    ]
}

# Turn the data into a DataFrame
df = pd.DataFrame(data)

print("Customer Data:")
print(df)


# -----------------------------
# 2. Explore the data
# -----------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())


# -----------------------------
# 3. Check for missing values
# -----------------------------

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# 4. Select features and target
# -----------------------------

# These are the columns that the model will use
# to make its prediction.

X = df[["Age", "Monthly_Charges", "Tenure"]]

# This is the value that we want the model to predict.
y = df["Churn"]


# -----------------------------
# 5. Split the data
# -----------------------------

# 80% of the data is used for training
# and 20% is used for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 6. Scale the features
# -----------------------------

# Scaling helps the model work better
# when the features have different ranges.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 7. Create the model
# -----------------------------

# I used Logistic Regression because
# this is a simple classification problem.

model = LogisticRegression()


# -----------------------------
# 8. Train the model
# -----------------------------

model.fit(X_train, y_train)

print("\nModel training is complete.")


# -----------------------------
# 9. Make predictions
# -----------------------------

# Now the model tries to predict
# which customers will leave.

y_pred = model.predict(X_test)


# -----------------------------
# 10. Evaluate the model
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nModel Results:")
print("Accuracy :", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall   :", round(recall, 2))
print("F1-Score :", round(f1, 2))


# -----------------------------
# 11. Classification Report
# -----------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------
# 12. Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# -----------------------------
# 13. Show the Confusion Matrix
# -----------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Add the numbers inside the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.xticks([0, 1], ["Stayed", "Churned"])
plt.yticks([0, 1], ["Stayed", "Churned"])

plt.colorbar()
plt.tight_layout()
plt.show()


# -----------------------------
# 14. Visualize customer churn
# -----------------------------

churn_count = df["Churn"].value_counts()

plt.figure(figsize=(6, 5))

plt.bar(
    ["Stayed", "Churned"],
    [
        churn_count.get(0, 0),
        churn_count.get(1, 0)
    ]
)

plt.title("Customer Churn")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# -----------------------------
# 15. Final message
# -----------------------------

print("\nProject completed successfully!")

print(
    "This project predicts whether a customer "
    "is likely to leave the company using "
    "Logistic Regression."
)
