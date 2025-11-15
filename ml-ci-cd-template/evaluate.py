import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_model(model_path="models/model.pkl", X_test_path="data/X_test.csv", y_test_path="data/y_test.csv"):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Error: Test data not found at {X_test_path} or {y_test_path}. Please train the model first to generate test data.")
        return

    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze() # .squeeze() to convert DataFrame to Series

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # You might want to save these metrics to a file or a database
    with open("models/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    print("Evaluation metrics saved to models/metrics.txt")

if __name__ == "__main__":
    evaluate_model()
