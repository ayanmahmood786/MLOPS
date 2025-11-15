# ML CI/CD Template

This repository provides a template for setting up a Continuous Integration/Continuous Deployment (CI/CD) pipeline for a Machine Learning project using GitHub Actions.

## Project Structure

```
ml-ci-cd-template/
├── .github/
│ └── workflows/
│ ├── train.yml       # GitHub Actions workflow for model training and registration
│ └── deploy.yml      # GitHub Actions workflow for model deployment
├── app/
│ └── main.py         # FastAPI application for model inference
├── models/           # Directory to store trained models (local during development)
├── train.py          # Script for training the ML model
├── evaluate.py       # Script for evaluating the trained model
├── register_model.py # Script for registering the model (e.g., to a model registry)
├── model_utils.py    # Utility functions for model loading and preprocessing
├── requirements.txt  # Python dependencies
├── Dockerfile        # Dockerfile for containerizing the FastAPI application
├── README.md         # Project README
└── .gitignore        # Git ignore file
```

## Getting Started

### 1. Local Development

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ml-ci-cd-template.git
    cd ml-ci-cd-template
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Train the model:**
    ```bash
    python train.py
    ```
    This will create a dummy dataset, train a `LogisticRegression` model, and save it to `models/model.pkl`. It will also save test data for evaluation.
4.  **Evaluate the model:**
    ```bash
    python evaluate.py
    ```
    This will load the trained model and test data, and print evaluation metrics.
5.  **Register the model:**
    ```bash
    python register_model.py
    ```
    This will copy the `model.pkl` to `models/registry/` with a timestamp and update a `latest_model.txt` pointer.
6.  **Run the FastAPI application locally:**
    ```bash
    python app/main.py
    ```
    The API will be available at `http://127.0.0.1:8000`. You can test the `/predict` endpoint using `curl` or a tool like Postman.

    Example `curl` command for prediction:
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"feature1": 5, "feature2": 6}'
    ```

### 2. Dockerization

To build and run the FastAPI application using Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t ml-api .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 ml-api
    ```
    The API will be available at `http://localhost:8000`.

### 3. GitHub Actions CI/CD

This template includes two GitHub Actions workflows:

*   `.github/workflows/train.yml`:
    *   **Triggered by:** Pushes to `main` branch affecting model-related files (`train.py`, `evaluate.py`, etc.) or manual `workflow_dispatch`.
    *   **Actions:** Sets up Python, installs dependencies, runs `train.py`, `evaluate.py`, and `register_model.py`.
*   `.github/workflows/deploy.yml`:
    *   **Triggered by:** Completion of the `Train Model` workflow or manual `workflow_dispatch`.
    *   **Actions:** Sets up Python, installs dependencies, and simulates a model deployment. **You will need to replace the placeholder deployment logic with your actual deployment steps** (e.g., deploying to a cloud ML platform, updating a Kubernetes service, etc.).

## Customization

*   **Model Training:** Modify `train.py` to use your actual dataset, model architecture, and training logic.
*   **Model Evaluation:** Update `evaluate.py` with relevant metrics and validation procedures for your specific problem.
*   **Model Registration:** Integrate `register_model.py` with a proper model registry solution (e.g., MLflow, Azure ML Model Registry, SageMaker Model Registry) for versioning and metadata management.
*   **FastAPI Application:** Adjust `app/main.py` to load your specific model and handle input/output according to your model's requirements.
*   **Deployment:** Crucially, update the `deploy.yml` workflow with your actual deployment strategy. This will vary greatly depending on your infrastructure (e.g., cloud provider, Kubernetes, serverless functions).
*   **Dependencies:** Update `requirements.txt` with all necessary Python packages.
