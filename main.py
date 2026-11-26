import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load dataset
df = pd.read_csv("dataset/Telco-Customer-Churn.csv")


# Data cleaning
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.drop("customerID", axis=1)
df = df.dropna()


# Separate X and y
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})


# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["str", "object"]).columns
numerical_columns = X.select_dtypes(include=["number"]).columns


# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", StandardScaler(), numerical_columns)
    ]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Preprocessing for model comparison
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Display preprocessing results
print("Original X shape:", X.shape)
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nPreprocessing completed successfully!")


# LOGISTIC REGRESSION
model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# DECISION TREE
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

dt_model.fit(X_train_processed, y_train)

dt_pred = dt_model.predict(X_test_processed)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print("\nClassification Report:")
print(classification_report(y_test, dt_pred))


# RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)

rf_model.fit(X_train_processed, y_train)

rf_pred = rf_model.predict(X_test_processed)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# FINAL MODEL: LOGISTIC REGRESSION WITH PIPELINE
final_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classification", LogisticRegression(max_iter=1000))
])

# Train using original, unprocessed data
final_model.fit(X_train, y_train)

# Evaluate final model
final_pred = final_model.predict(X_test)

print("\nFinal Model Results")
print("Accuracy:", accuracy_score(y_test, final_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))
print("\nClassification Report:")
print(classification_report(y_test, final_pred))


# Save complete pipeline
joblib.dump(final_model, "churn_model.pkl")

print("\nModel saved successfully as churn_model.pkl!")