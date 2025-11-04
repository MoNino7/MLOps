import argparse
import pickle
import mlflow
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Configurable Variables ---
TRAINING_ITERATIONS = 1000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    """Loads the Heart Disease dataset."""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets.squeeze()  # Ensure y is a 1D array
    # In the 'num' column, 0 means no heart disease and 1, 2, 3, 4 mean heart disease.
    # We will convert this to a binary classification problem.
    y = y.apply(lambda x: 1 if x > 0 else 0)
    return X, y


def train_and_evaluate(X, y, model_type, C=1.0, max_depth=None):
    """Trains and evaluates a specified model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Drop rows with NaNs
    X_train_no_nan = X_train.dropna()
    y_train_no_nan = y_train.loc[X_train_no_nan.index]
    X_test_no_nan = X_test.dropna()
    y_test_no_nan = y_test.loc[X_test_no_nan.index]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_no_nan)
    X_test_scaled = scaler.transform(X_test_no_nan)

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("training_iterations", TRAINING_ITERATIONS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        if model_type == "LogisticRegression":
            mlflow.log_param("C", C)
            model = LogisticRegression(
                max_iter=TRAINING_ITERATIONS, C=C, random_state=RANDOM_STATE
            )
        elif model_type == "DecisionTreeClassifier":
            mlflow.log_param("max_depth", max_depth)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
        else:
            raise ValueError("Unsupported model type")

        model.fit(X_train_scaled, y_train_no_nan)

        # Evaluate the model
        test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test_no_nan, test_pred)
        train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train_no_nan, train_pred)

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_accuracy", train_acc)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"--- {model_type} Results (C={C}, max_depth={max_depth}) ---")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print("--------------------------------")


def main():
    """Main function to load data, train models with different hyperparameters, and log them."""
    C_values = [0.01, 0.1, 1.0, 10, 100]
    depth_values = [2, 5, 10, 20, None]

    X, y = load_data()

    # Train Logistic Regression models
    for c in C_values:
        print(f"--- Training LogisticRegression with C={c} ---")
        train_and_evaluate(X, y, "LogisticRegression", C=c)

    # Train Decision Tree models
    for depth in depth_values:
        print(f"--- Training DecisionTreeClassifier with max_depth={depth} ---")
        train_and_evaluate(X, y, "DecisionTreeClassifier", max_depth=depth)

    print("\n--- All training runs completed ---")


if __name__ == "__main__":
    main()
