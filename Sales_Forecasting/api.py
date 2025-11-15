import os
import subprocess
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger

# --------------------------
# 1️⃣ Setup log directory
# --------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger.add(f"{LOG_DIR}/app.log", rotation="10 MB", retention="7 days", enqueue=True)
logger.info("Logger initialized")

# --------------------------
# 2️⃣ Start log-viewer on port 9000
# --------------------------
try:
    subprocess.Popen(["log-viewer", LOG_DIR, "-p", "9000"])
    logger.info("Log-viewer started on http://localhost:9000")
except FileNotFoundError:
    logger.error("log-viewer is not installed or not in PATH. Install via `pip install log-viewer`")

# --------------------------
# 3️⃣ Create FastAPI app
# --------------------------
app = FastAPI(title="EVA API with Log Viewer")

# --------------------------
# 4️⃣ Example route
# --------------------------
@app.get("/api/hello")
async def hello():
    logger.info("Hello endpoint called")
    return {"message": "Hello, EVA API!"}

# --------------------------
# 5️⃣ Redirect to log-viewer
# --------------------------
@app.get("/logs")
async def logs():
    return RedirectResponse("http://localhost:9000/")

# --------------------------
# 6️⃣ Optional startup event
# --------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application started")
