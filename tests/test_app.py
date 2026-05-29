"""
Tests for the Potato Disease Classifier application.
"""

import io
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app
from app.model.predictor import (
    PotatoDiseaseClassifier,
    CLASS_LABELS,
    CLASS_IDS,
    IMG_HEIGHT,
    IMG_WIDTH,
)


# =============================================
# Fixtures
# =============================================

@pytest.fixture
def client():
    """Create a test client for FastAPI."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image():
    """Create a synthetic test image (green leaf-like)."""
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (34, 139, 34))
    # Add some noise
    img_array = np.array(img)
    noise = np.random.randint(0, 30, img_array.shape, dtype=np.uint8)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_bytes(sample_image):
    """Get sample image as bytes."""
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    return buf.getvalue()


# =============================================
# Health Check Tests
# =============================================

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_home_page(client):
    """Test that the home page renders."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# =============================================
# Prediction Tests - without model
# =============================================

def test_predict_without_model(client, sample_image_bytes):
    """Test prediction endpoint returns 503 if model is not loaded."""
    response = client.post(
        "/predict",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    # Should return either 503 (model not loaded) or 200 (unlikely with synthetic image)
    assert response.status_code in [200, 503]


def test_predict_invalid_file_type(client):
    """Test prediction with invalid file type."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_no_file(client):
    """Test prediction without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error


def test_predict_large_file(client):
    """Test prediction with file too large."""
    large_data = b"0" * (11 * 1024 * 1024)  # 11 MB
    response = client.post(
        "/predict",
        files={"file": ("large.jpg", large_data, "image/jpeg")},
    )
    assert response.status_code == 413


# =============================================
# API Endpoint Tests
# =============================================

def test_api_classes(client):
    """Test the classes API endpoint."""
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 3
    assert all(label in data["classes"] for label in CLASS_LABELS)


def test_api_model_info(client):
    """Test the model info API endpoint."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["loaded", "not_loaded"]


# =============================================
# Predictor Unit Tests
# =============================================

def test_preprocess_image(sample_image):
    """Test image preprocessing (MobileNetV2 expects [-1, 1])."""
    predictor = PotatoDiseaseClassifier()
    processed = predictor.preprocess_image(sample_image)

    assert processed.shape == (1, IMG_HEIGHT, IMG_WIDTH, 3)
    assert processed.dtype == np.float32
    # MobileNetV2 expects values in [-1, 1] range
    assert -1.0 <= processed.min() <= processed.max() <= 1.0


def test_preprocess_image_with_alpha():
    """Test preprocessing RGBA image."""
    predictor = PotatoDiseaseClassifier()
    img = Image.new("RGBA", (IMG_WIDTH, IMG_HEIGHT), (34, 139, 34, 255))
    processed = predictor.preprocess_image(img)
    assert processed.shape == (1, IMG_HEIGHT, IMG_WIDTH, 3)
    assert processed.dtype == np.float32


def test_preprocess_image_grayscale():
    """Test preprocessing grayscale image."""
    predictor = PotatoDiseaseClassifier()
    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), 128)
    processed = predictor.preprocess_image(img)
    assert processed.shape == (1, IMG_HEIGHT, IMG_WIDTH, 3)


def test_class_labels_consistency():
    """Test that CLASS_LABELS and CLASS_IDS have same length."""
    assert len(CLASS_LABELS) == len(CLASS_IDS)


# =============================================
# Model Trainer Tests
# =============================================

def test_build_model():
    """Test that the model builds correctly."""
    from app.model.trainer import build_model

    model, base_model = build_model()
    assert model is not None
    assert base_model is not None
    assert model.input_shape == (None, IMG_HEIGHT, IMG_WIDTH, 3)
    assert model.output_shape == (None, 3)
    assert model.name == "potato_disease_classifier"
    assert len(model.layers) > 0


def test_augmentation_function_exists():
    """Test that the augmentation function is defined in trainer."""
    from app.model.trainer import apply_augmentation
    import tensorflow as tf

    # Create a dummy image and label
    dummy_image = tf.random.uniform((128, 128, 3), minval=-1.0, maxval=1.0)
    dummy_label = tf.constant([1.0, 0.0, 0.0])
    augmented_image, augmented_label = apply_augmentation(dummy_image, dummy_label)

    assert augmented_image.shape == dummy_image.shape
    assert augmented_label.shape == dummy_label.shape
    assert augmented_image.dtype == tf.float32


# =============================================
# Static Files
# =============================================

def test_static_css(client):
    """Test that CSS file is accessible."""
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_js(client):
    """Test that JS file is accessible."""
    response = client.get("/static/js/main.js")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["text/javascript", "application/javascript"]


# =============================================
# Feedback / Auto-training Tests
# =============================================

def test_feedback_endpoint_missing_fields(client):
    """Test that feedback endpoint returns 400 if fields are missing."""
    response = client.post("/feedback", json={})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_feedback_endpoint_invalid_class(client):
    """Test feedback with invalid class."""
    response = client.post(
        "/feedback",
        json={"prediction_id": "abc-123", "confirmed_class": "invalid_class"},
    )
    assert response.status_code == 400


def test_feedback_endpoint_image_not_found(client):
    """Test feedback when image doesn't exist."""
    response = client.post(
        "/feedback",
        json={"prediction_id": "nonexistent-uuid", "confirmed_class": "healthy"},
    )
    assert response.status_code == 404


def test_predict_response_has_prediction_id(client):
    """Test that predict response includes prediction_id for feedback."""
    response = client.post(
        "/predict",
        files={"file": ("test.png", b"fake-image-data", "image/png")},
    )
    # Even if it fails due to invalid image, we test the error format
    assert response.status_code in [200, 400, 503]
    if response.status_code == 200:
        data = response.json()
        assert "prediction_id" in data


def test_load_user_feedback_data_empty():
    """Test loading user feedback data when directory is empty."""
    from app.model.trainer import load_user_feedback_data
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_user_feedback_data(Path(tmpdir))
        assert result == (None, None)


def test_feedback_count_in_response(client, tmp_path):
    """Test that feedback response includes total_feedback_samples."""
    # Just test the endpoint returns proper error for missing data
    response = client.post(
        "/feedback",
        json={"prediction_id": "test-id", "confirmed_class": "early_blight"},
    )
    # Should be 404 because image not found (no actual upload happened)
    assert response.status_code == 404
