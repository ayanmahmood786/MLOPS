import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(data_path="data/sample_data.csv"):
    # Create a dummy data directory and sample data if they don't exist
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(data_path):
        print(f"Creating dummy data at {data_path}")
        dummy_data = {
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        }
        pd.DataFrame(dummy_data).to_csv(data_path, index=False)

    df = pd.read_csv(data_path)
    X = df[['feature1', 'feature2']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model trained and saved to models/model.pkl")

    # For demonstration, let's also save the test set for evaluation
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    print("Test data saved to data/X_test.csv and data/y_test.csv")

if __name__ == "__main__":
    train_model()
