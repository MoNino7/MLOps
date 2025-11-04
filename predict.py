
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# --- Configurable Variables ---
MODEL_PATH = "model.pkl"

def load_model(path):
    """Loads a model from a pickle file."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model

def preprocess_data(X):
    """Preprocesses new data using the same steps as training."""
    # In a real-world scenario, the scaler would also be saved and loaded.
    # For this example, we'll re-initialize and fit a scaler on the full dataset
    # to simulate the training environment's scaling.
    # This is NOT ideal for production; a full pipeline should be saved.

    # Fetch the full dataset to fit the scaler correctly
    heart_disease = fetch_ucirepo(id=45)
    X_full = heart_disease.data.features

    # Drop rows with NaNs (consistent with training)
    X_no_nan = X.dropna()

    # Fit and transform scaler (this should ideally be the *saved* scaler from training)
    scaler = StandardScaler()
    scaler.fit(X_full.dropna()) # Fit on the full, cleaned dataset
    X_scaled = scaler.transform(X_no_nan)

    return X_scaled, X_no_nan.index # Return index to map back if needed

def main():
    """Main function to load model and make predictions."""
    model = load_model(MODEL_PATH)

    # Example: Create some dummy data for prediction
    # In a real scenario, this would come from a new source
    # For demonstration, let's take the first row of the original dataset
    heart_disease = fetch_ucirepo(id=45)
    X_original = heart_disease.data.features
    sample_data = X_original.iloc[[0]] # Take the first sample as a DataFrame

    print("\nOriginal sample data for prediction:")
    print(sample_data)

    # Preprocess the sample data
    processed_sample_data, original_indices = preprocess_data(sample_data)

    # Make prediction
    prediction = model.predict(processed_sample_data)

    print(f"\nPrediction for the sample data (original index {original_indices.tolist()[0]}): {prediction[0]}")

if __name__ == "__main__":
    main()
