import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    df = pd.read_csv("dataset.csv")

    # Target: churn Yes/No
    y = df["churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["churn"])

    categorical_cols = ["contract_type", "payment_method", "internet_service", "tech_support"]
    numeric_cols = ["tenure_months", "monthly_charges", "total_charges", "num_products", "is_senior", "has_partner"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = LogisticRegression(max_iter=200)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, "churn_model.joblib")
    print("\nSaved model: churn_model.joblib")


if __name__ == "__main__":
    main()
