import joblib
import os
import datetime

def register_model(model_path="models/model.pkl", registry_dir="models/registry"):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    os.makedirs(registry_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    registered_model_name = f"model_{timestamp}.pkl"
    registered_model_path = os.path.join(registry_dir, registered_model_name)

    # Copy the model to the registry
    model = joblib.load(model_path)
    joblib.dump(model, registered_model_path)

    print(f"Model registered successfully at {registered_model_path}")

    # In a real scenario, you would update a database or a manifest file
    # to keep track of registered models, their versions, and metadata.
    with open(os.path.join(registry_dir, "latest_model.txt"), "w") as f:
        f.write(registered_model_name)
    print(f"Updated latest model pointer to {registered_model_name}")

if __name__ == "__main__":
    register_model()
