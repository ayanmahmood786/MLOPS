import joblib
import os

def load_latest_model(registry_dir="models/registry"):
    """
    Loads the latest registered model from the model registry.
    In a real scenario, this would involve more robust versioning and metadata.
    """
    latest_model_pointer_path = os.path.join(registry_dir, "latest_model.txt")
    if not os.path.exists(latest_model_pointer_path):
        print(f"Error: No latest model pointer found in {registry_dir}.")
        return None

    with open(latest_model_pointer_path, "r") as f:
        latest_model_name = f.read().strip()

    model_path = os.path.join(registry_dir, latest_model_name)
    if not os.path.exists(model_path):
        print(f"Error: Latest model file not found at {model_path}.")
        return None

    model = joblib.load(model_path)
    print(f"Successfully loaded model: {latest_model_name}")
    return model

def preprocess_data(data):
    """
    Dummy preprocessing function. Replace with actual preprocessing logic.
    """
    # Example: Convert dictionary to DataFrame, handle missing values, scale features, etc.
    # For this template, we assume the input data is already in a suitable format
    # or that the model expects raw features.
    print("Performing dummy data preprocessing.")
    return data

if __name__ == "__main__":
    # Example usage:
    # This part would typically be used by the app/main.py or other inference services
    # to load the model.
    print("Attempting to load the latest model:")
    loaded_model = load_latest_model()
    if loaded_model:
        print("Model loaded successfully for utility testing.")
        # You can add a dummy prediction here to test
        # dummy_input = pd.DataFrame([[1, 2]], columns=['feature1', 'feature2'])
        # preprocessed_input = preprocess_data(dummy_input)
        # prediction = loaded_model.predict(preprocessed_input)
        # print(f"Dummy prediction: {prediction}")
    else:
        print("Failed to load model for utility testing.")
