
import io
import os
import time
import logging
from collections import deque
from threading import Lock

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("cifar10-api")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "cifar10_model.keras")
)

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "32"))
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

NORMALIZE_INPUT = os.getenv(
    "NORMALIZE_INPUT", "0"
).lower() in ("1", "true", "yes")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# --------------------------------------------------
# 2. Model loading
# --------------------------------------------------

model = None
model_lock = Lock()

request_count = 0
error_count = 0
recent_latencies = deque(maxlen=500)
metrics_lock = Lock()


def load_cifar_model():
    """Load the trained Keras model."""
    global model

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place cifar10_model.keras in the backend/model folder."
        )

    loaded_model = load_model(MODEL_PATH, compile=False)

    # CIFAR-10 output must contain ten class scores.
    output_shape = loaded_model.output_shape
    if isinstance(output_shape, list):
        raise ValueError("This API expects a single-output classification model.")

    if output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected 10 output classes, got {output_shape}."
        )

    model = loaded_model

    logger.info("Model loaded successfully: %s", MODEL_PATH)
    logger.info("Model input shape: %s", model.input_shape)
    logger.info("Model output shape: %s", model.output_shape)


# --------------------------------------------------
# 3. Image preprocessing
# --------------------------------------------------

def preprocess_image(image_bytes):
    """Convert uploaded image to a batch suitable for the model."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32)

    # Enable only if this matches the model's training preprocessing.
    if NORMALIZE_INPUT:
        array = array / 255.0

    return np.expand_dims(array, axis=0)


# --------------------------------------------------
# 4. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "project": "Task 15 - End-to-End Deep Learning Deployment",
        "service": "CIFAR-10 Image Classification API",
        "model_loaded": model is not None,
        "endpoints": {
            "home": "GET /",
            "health": "GET /health",
            "prediction": "POST /predict",
            "metrics": "GET /metrics"
        }
    })


# --------------------------------------------------
# 5. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    if model is None:
        return jsonify({
            "status": "not_ready",
            "model_loaded": False
        }), 503

    return jsonify({
        "status": "healthy",
        "model_loaded": True
    }), 200


# --------------------------------------------------
# 6. Image prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    global request_count, error_count

    started = time.perf_counter()

    with metrics_lock:
        request_count += 1

    try:
        if model is None:
            raise RuntimeError("The classification model is not loaded.")

        if "file" not in request.files:
            with metrics_lock:
                error_count += 1

            return jsonify({
                "error": "Missing image file. Upload it using the 'file' field."
            }), 400

        uploaded_file = request.files["file"]
        image_bytes = uploaded_file.read()

        if not image_bytes:
            with metrics_lock:
                error_count += 1

            return jsonify({
                "error": "The uploaded file is empty."
            }), 400

        # Decode and preprocess the image.
        try:
            image_array = preprocess_image(image_bytes)
        except (UnidentifiedImageError, OSError, ValueError):
            with metrics_lock:
                error_count += 1

            return jsonify({
                "error": "Invalid image. Upload a valid PNG, JPEG, or supported image."
            }), 400

        # Protect model inference against concurrent calls.
        with model_lock:
            prediction = np.asarray(
                model.predict(image_array, verbose=0)
            )

        scores = prediction[0].astype(np.float64)

        if scores.shape != (len(CLASS_NAMES),):
            raise ValueError(
                f"Unexpected model output shape: {prediction.shape}"
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError("Model returned non-finite prediction scores.")

        # Convert logits to probabilities when needed.
        # If the model already returns probabilities, preserve them.
        if (
            np.any(scores < 0)
            or np.any(scores > 1)
            or not np.isclose(np.sum(scores), 1.0, atol=0.05)
        ):
            shifted = scores - np.max(scores)
            exponentials = np.exp(shifted)
            scores = exponentials / np.sum(exponentials)

        class_index = int(np.argmax(scores))
        elapsed_ms = (time.perf_counter() - started) * 1000

        with metrics_lock:
            recent_latencies.append(elapsed_ms)

        logger.info(
            "Prediction completed: class=%s latency_ms=%.2f",
            CLASS_NAMES[class_index],
            elapsed_ms
        )

        return jsonify({
            "predicted_class": CLASS_NAMES[class_index],
            "class_index": class_index,
            "confidence": round(float(scores[class_index]), 6),
            "probabilities": {
                CLASS_NAMES[i]: round(float(scores[i]), 6)
                for i in range(len(CLASS_NAMES))
            },
            "inference_time_ms": round(elapsed_ms, 2)
        }), 200

    except Exception:
        with metrics_lock:
            error_count += 1

        logger.exception("Prediction request failed")

        return jsonify({
            "error": "Prediction failed. Check backend logs for details."
        }), 500


# --------------------------------------------------
# 7. Runtime metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    with metrics_lock:
        latencies = list(recent_latencies)
        total_requests = request_count
        total_errors = error_count

    average_latency = (
        float(np.mean(latencies)) if latencies else 0.0
    )

    p95_latency = (
        float(np.percentile(latencies, 95)) if latencies else 0.0
    )

    return jsonify({
        "requests_total": total_requests,
        "errors_total": total_errors,
        "successful_inferences": len(latencies),
        "average_latency_ms": round(average_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "latency_samples": len(latencies)
    })


# --------------------------------------------------
# 8. Request size error handler
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "Uploaded request exceeds the 8 MB limit."
    }), 413


# --------------------------------------------------
# 9. Load model and start server
# --------------------------------------------------

# Load at module import so Gunicorn's app:app entry point works.
load_cifar_model()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
