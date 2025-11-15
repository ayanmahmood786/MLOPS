from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd

app = FastAPI()

# Load the model (this should be replaced with a model registry lookup in a real scenario)
model = None
try:
    model = joblib.load("models/model.pkl")
except FileNotFoundError:
    print("Model not found. Please train and register a model first.")

@app.get("/")
def read_root():
    return {"message": "ML Model API is running!"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded. Please train and register a model."}

    try:
        # Assuming data is a dictionary that can be converted to a pandas DataFrame
        # You might need more sophisticated data validation and preprocessing here
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df).tolist()
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
