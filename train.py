import os
import pickle
import mlflow
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- Global Config ---
TRAINING_ITERATIONS = 1000
TEST_SIZE = 0.2
RANDOM_STATE = 42
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "heart_model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.txt")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def data_preparation():
    """Load and prepare the UCI Heart Disease dataset."""
    try:
        print("Loading Heart Disease dataset...")
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets.squeeze()
        y = y.apply(lambda x: 1 if x > 0 else 0)

        print(f"Dataset loaded: X shape = {X.shape}, y shape = {y.shape}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Drop NaN rows
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Data preparation complete.")
        return X_train_scaled, X_test_scaled, y_train, y_test

    except Exception as e:
        print(f"[Error] Data preparation failed: {e}")
        raise


def train(X_train, y_train, model_type="LogisticRegression", C=1.0, max_depth=None):
    """Train the specified model."""
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("heart_disease_classification")

        with mlflow.start_run():
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("training_iterations", TRAINING_ITERATIONS)
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

            print(f"Training {model_type} model...")
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")

            print("Model training complete.")
            return model

    except Exception as e:
        print(f"[Error] Training failed: {e}")
        raise


def evaluate(model, X_train, y_train, X_test, y_test):
    """Evaluate the trained model and save metrics."""
    try:
        print("Evaluating model...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        # Print results
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")

        # Save metrics to file
        with open(METRICS_PATH, "w") as f:
            f.write(f"Train Accuracy: {train_acc:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        print(f"Metrics saved to {METRICS_PATH}")

        return {"train_acc": train_acc, "test_acc": test_acc}

    except Exception as e:
        print(f"[Error] Evaluation failed: {e}")
        raise


def save_model(model, path):
    """Save trained model to pickle."""
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"[Error] Failed to save model: {e}")
        raise


def main():
    X_train, X_test, y_train, y_test = data_preparation()
    model = train(X_train, y_train, model_type="LogisticRegression", C=1.0)
    metrics = evaluate(model, X_train, y_train, X_test, y_test)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
