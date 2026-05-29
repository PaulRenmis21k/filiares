"""
Potato Disease Classifier - Model Predictor
============================================
Carga el modelo entrenado y realiza predicciones sobre nuevas imágenes.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only

import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = Path("models/potato_disease_model.keras")

CLASS_LABELS = ["Early Blight", "Late Blight", "Healthy"]
CLASS_IDS = ["early_blight", "late_blight", "healthy"]

# Color mapping for each class (for visual feedback)
CLASS_COLORS = {
    "Early Blight": "#e67e22",  # Orange
    "Late Blight": "#e74c3c",  # Red
    "Healthy": "#27ae60",  # Green
}


class PotatoDiseaseClassifier:
    """Singleton class for loading and using the potato disease classification model."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self) -> bool:
        """Load the trained model from disk."""
        try:
            if not MODEL_PATH.exists():
                print(f"[WARNING] Model not found at {MODEL_PATH}")
                return False

            self._model = keras.models.load_model(str(MODEL_PATH))
            print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for model inference.

        Steps:
            1. Convert to RGB (remove alpha channel if present)
            2. Resize to target dimensions
            3. Normalize pixel values to [0, 1]
            4. Add batch dimension
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert to array and normalize to [-1, 1] (MobileNetV2 expects this range)
        img_array = np.array(image, dtype=np.float32) / 127.5 - 1.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(
        self, image: Image.Image
    ) -> Tuple[str, str, float, List[float]]:
        """
        Predict the disease class for a given image.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (class_name, class_id, confidence, all_probabilities)
        """
        if self._model is None:
            loaded = self.load_model()
            if not loaded:
                raise RuntimeError(
                    "Model not loaded. Please train the model first using "
                    "'python -m app.model.trainer'"
                )

        # Preprocess
        processed = self.preprocess_image(image)

        # Predict
        predictions = self._model.predict(processed, verbose=0)[0]

        # Get class with highest probability
        predicted_class_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_idx])

        class_name = CLASS_LABELS[predicted_class_idx]
        class_id = CLASS_IDS[predicted_class_idx]

        return class_name, class_id, confidence, predictions.tolist()

    def get_top_k_predictions(
        self, image: Image.Image, k: int = 3
    ) -> List[dict]:
        """Get top-k predictions with confidence scores."""
        _, _, _, all_probs = self.predict(image)

        # Get top-k indices
        probs_array = np.array(all_probs)
        top_k_indices = np.argsort(probs_array)[::-1][:k]

        results = []
        for idx in top_k_indices:
            results.append(
                {
                    "class_name": CLASS_LABELS[idx],
                    "class_id": CLASS_IDS[idx],
                    "confidence": float(probs_array[idx]),
                    "color": CLASS_COLORS[CLASS_LABELS[idx]],
                }
            )

        return results

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self._model is None:
            return {"status": "not_loaded"}

        total_params = self._model.count_params()
        trainable_params = sum(
            tf.size(w).numpy()
            for w in self._model.trainable_weights
        )

        return {
            "status": "loaded",
            "name": self._model.name,
            "input_shape": str(self._model.input_shape),
            "output_shape": str(self._model.output_shape),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "layers": len(self._model.layers),
            "classes": CLASS_LABELS,
        }


# Global instance for reuse
classifier = PotatoDiseaseClassifier()
