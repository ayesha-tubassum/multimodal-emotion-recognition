"""FastAPI application for multimodal (text, audio, image) emotion inference."""

from __future__ import annotations

import io
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fusion_model.keras"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"

# Input shapes and order match models/fusion_model.keras.
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
TEXT_SEQUENCE_LENGTH = 30
FACE_IMAGE_SIZE = (48, 48)
MFCC_COEFFICIENTS = 40
MFCC_FRAMES = 130
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the model once when the server begins accepting requests."""
    application.state.model = None
    application.state.model_error = None
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file was not found at {MODEL_PATH}")
        application.state.model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Fusion model loaded from %s", MODEL_PATH)
    except Exception as exc:  # Surface a friendly page while preserving diagnostics in logs.
        application.state.model_error = str(exc)
        logger.exception("Unable to load fusion model")
    yield


app = FastAPI(title="AffectSense | Multimodal Emotion Recognition", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


def preprocess_text(text: str) -> np.ndarray:
    """Create a padded sequence of length 30.

    This lightweight hashing tokenizer is a deployment-safe stub. Replace it with
    the exact tokenizer/vocabulary used at training time for production accuracy.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    token_ids = [((sum(ord(char) for char in token) % 19_999) + 1) for token in tokens]
    token_ids = token_ids[:TEXT_SEQUENCE_LENGTH]
    padded = np.zeros(TEXT_SEQUENCE_LENGTH, dtype=np.int32)
    padded[: len(token_ids)] = token_ids
    return padded[np.newaxis, :]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode a face image into the model's (1, 48, 48, 1) grayscale tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded face file is not a valid image.") from exc
    resized = image.resize(FACE_IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array[np.newaxis, ..., np.newaxis]


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode WAV audio and pad/truncate MFCCs to the model's (1, 130, 40) input."""
    try:
        signal, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=22_050, mono=True)
    except Exception as exc:
        raise ValueError("The uploaded audio file could not be decoded. Please use a WAV file.") from exc
    if signal.size == 0:
        raise ValueError("The uploaded audio file is empty.")
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=MFCC_COEFFICIENTS)
    padded = np.zeros((MFCC_FRAMES, MFCC_COEFFICIENTS), dtype=np.float32)
    frame_count = min(mfcc.shape[1], MFCC_FRAMES)
    padded[:frame_count, :] = mfcc[:, :frame_count].T
    return padded[np.newaxis, ...]


def page_context(**values: object) -> dict[str, object]:
    return {"prediction": None, "confidence": None, "error": None, "submitted_text": "", **values}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", page_context())


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...),
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
) -> HTMLResponse:
    context = page_context(submitted_text=text)
    if request.app.state.model is None:
        context["error"] = "The model is unavailable. Place fusion_model.keras in the models directory and restart the service."
        return templates.TemplateResponse(request, "index.html", context, status_code=503)
    if not text.strip():
        context["error"] = "Please enter a short text sample."
        return templates.TemplateResponse(request, "index.html", context, status_code=422)
    if not (image.content_type or "").startswith("image/"):
        context["error"] = "Please upload a valid face image."
        return templates.TemplateResponse(request, "index.html", context, status_code=422)
    if audio.filename and not audio.filename.lower().endswith(".wav"):
        context["error"] = "Please upload audio in WAV format."
        return templates.TemplateResponse(request, "index.html", context, status_code=422)

    try:
        text_input = preprocess_text(text)
        audio_input = preprocess_audio(await audio.read())
        image_input = preprocess_image(await image.read())
        probabilities = np.asarray(request.app.state.model.predict([image_input, audio_input, text_input], verbose=0))[0]
        class_index = int(np.argmax(probabilities))
        label = EMOTION_LABELS[class_index] if class_index < len(EMOTION_LABELS) else f"Class {class_index}"
        context.update(prediction=label, confidence=round(float(probabilities[class_index]) * 100, 2))
    except (ValueError, OSError) as exc:
        context["error"] = str(exc)
        return templates.TemplateResponse(request, "index.html", context, status_code=422)
    except Exception:
        logger.exception("Prediction failed")
        context["error"] = "Prediction could not be completed. Verify that preprocessing matches your trained model inputs."
        return templates.TemplateResponse(request, "index.html", context, status_code=500)
    finally:
        await image.close()
        await audio.close()

    return templates.TemplateResponse(request, "index.html", context)
