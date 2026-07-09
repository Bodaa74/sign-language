"""
ASL Sign Language Translator - FastAPI Backend
===============================================
REST API for model inference.

Endpoints:
  POST /predict          — Classify a single hand image
  POST /predict/stream   — WebSocket for real-time video frames
  GET  /health           — Health check
  GET  /labels           — Return all class labels

Run:
  uvicorn backend.app.main:app --reload --port 8000
"""

import io
import json
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf

from backend.utils.predictor import ASLPredictor
from backend.utils.smoother import WebSocketSmoother

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Model Lifecycle
# ─────────────────────────────────────────────
MODEL_PATH = Path("model/saved/asl_model_final.keras")
LABEL_PATH = Path("model/saved/label_map.json")

predictor: ASLPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading ASL model …")
    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at {MODEL_PATH}. Prediction endpoints will fail.")
    else:
        predictor = ASLPredictor(str(MODEL_PATH), str(LABEL_PATH))
        logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down …")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="ASL Sign Language Translator API",
    description="Real-time American Sign Language A–Z recognition",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class PredictResponse(BaseModel):
    letter: str
    confidence: float
    top3: list[dict]   # [{'letter': 'A', 'confidence': 0.98}, ...]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes → BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image.")
    return img


def decode_base64_frame(b64: str) -> np.ndarray:
    """Decode base64 data-URL or raw base64 → BGR numpy array."""
    if "," in b64:
        b64 = b64.split(",")[1]
    raw = base64.b64decode(b64)
    return decode_image(raw)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "num_classes": predictor.num_classes if predictor else 0,
    }


@app.get("/labels")
async def get_labels():
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    return {"labels": predictor.labels}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accept a JPEG/PNG image upload and return the predicted ASL letter.
    """
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    try:
        contents = await file.read()
        img = decode_image(contents)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    result = predictor.predict(img)
    return result


@app.post("/predict/base64", response_model=PredictResponse)
async def predict_base64(payload: dict):
    """
    Accept a base64-encoded image (from webcam canvas) and return prediction.
    Body: {"image": "<base64 string>"}
    """
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    b64 = payload.get("image", "")
    if not b64:
        raise HTTPException(400, "Missing 'image' field")

    try:
        img = decode_base64_frame(b64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}")

    result = predictor.predict(img)
    return result


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frame prediction.
    Client sends: JSON {"image": "<base64>"}
    Server sends: JSON {"letter": "A", "confidence": 0.97, "top3": [...], "word": "..."}
    """
    await websocket.accept()
    smoother = WebSocketSmoother()
    word = ""
    last_stable = ""
    last_appended = ""
    stable_count = 0
    HOLD_FRAMES = 18

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            # Handle control messages
            if payload.get("type") == "control":
                action = payload.get("action")
                if action == "space":
                    word += " "
                    last_stable = ""
                    last_appended = ""
                elif action == "backspace":
                    word = word[:-1]
                    last_stable = ""
                    last_appended = word[-1] if word else ""
                elif action == "clear":
                    word = ""
                    last_stable = ""
                    last_appended = ""
                await websocket.send_text(json.dumps({"word": word}))
                continue

            b64 = payload.get("image", "")
            if not b64:
                continue

            if predictor is None:
                await websocket.send_text(json.dumps({"error": "Model not loaded"}))
                continue

            try:
                img = decode_base64_frame(b64)
            except Exception:
                continue

            raw_result = predictor.predict(img)

            if not raw_result.get("hand_detected", False):
                smoother.reset()
                last_stable = ""
                last_appended = ""
                stable_count = 0
                await websocket.send_text(json.dumps({
                    "letter": "",
                    "confidence": 0,
                    "stability": 0,
                    "top3": raw_result["top3"],
                    "candidate_letter": "",
                    "candidate_confidence": 0,
                    "hand_detected": False,
                    "word": word,
                }))
                continue

            letter, stability = smoother.update(raw_result["letter"], raw_result["confidence"])

            # Stability-based auto-append
            if letter and letter == last_stable:
                stable_count += 1
            else:
                stable_count = 1 if letter else 0
                last_stable = letter
                if not letter:
                    last_appended = ""

            if letter and stable_count == HOLD_FRAMES and letter != last_appended:
                word += letter
                last_appended = letter

            response = {
                "letter":     letter,
                "confidence": raw_result["confidence"],
                "stability":  round(stability, 4),
                "top3":       raw_result["top3"],
                "candidate_letter": raw_result.get("candidate_letter", ""),
                "candidate_confidence": raw_result.get("candidate_confidence", 0),
                "hand_detected": raw_result.get("hand_detected", False),
                "word":       word,
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
