"""
Potato Disease Classifier - FastAPI Application
================================================
Aplicación web para clasificar enfermedades en hojas de papa
usando un modelo CNN entrenado con TensorFlow/Keras.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

from app.model.predictor import classifier, CLASS_LABELS, CLASS_IDS

# -------------------- Configuration --------------------
# Las rutas pueden ser configuradas por variables de entorno
# (útil para Render.com con disco persistente)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "training_data"))
for class_id in CLASS_IDS:
    (TRAINING_DATA_DIR / class_id).mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# -------------------- App Initialization --------------------
app = FastAPI(
    title="Potato Disease Classifier",
    description="API para clasificar enfermedades en hojas de papa usando deep learning",
    version="1.0.0",
    contact={
        "name": "Team Potato",
        "url": "https://github.com/",
    },
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")


# -------------------- Startup / Shutdown --------------------
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = classifier.load_model()
    if not success:
        print("[WARNING] Model not found at startup. Please train the model.")
    else:
        info = classifier.get_model_info()
        print(f"[INFO] {info['name']} loaded successfully")
        print(f"[INFO] Total parameters: {info['total_params']:,}")


# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    model_info = classifier.get_model_info()
    is_loaded = model_info["status"] == "loaded"
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_loaded": is_loaded,
            "classes": CLASS_LABELS,
            "model_info_json": str(model_info),
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    model_info = classifier.get_model_info()
    return {
        "status": "healthy" if model_info["status"] == "loaded" else "degraded",
        "model_loaded": model_info["status"] == "loaded",
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict potato disease from an uploaded leaf image.

    Returns the top prediction with confidence scores for all classes.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file contents
    contents = await file.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB",
        )

    try:
        # Open image
        image = Image.open(io.BytesIO(contents))

        # Validate it's a valid image
        image.verify()
        image = Image.open(io.BytesIO(contents))  # Reopen after verify

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Check if model is loaded
    if not classifier.is_loaded():
        loaded = classifier.load_model()
        if not loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please train the model first.",
            )

    try:
        # Get predictions
        predictions = classifier.get_top_k_predictions(image)

        # Save uploaded file for record
        prediction_id = str(uuid.uuid4())
        unique_filename = f"{prediction_id}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        image.save(file_path, optimize=True)

        return {
            "success": True,
            "prediction_id": prediction_id,
            "filename": file.filename,
            "predictions": predictions,
            "top_prediction": predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request, result: Optional[str] = None):
    """Render prediction result page (for GET redirects)."""
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result,
        },
    )


@app.get("/api/model-info")
async def api_model_info():
    """API endpoint to get model information."""
    return classifier.get_model_info()


@app.get("/api/classes")
async def api_classes():
    """API endpoint to get available classes."""
    return {"classes": CLASS_LABELS}


@app.post("/feedback")
async def receive_feedback(data: dict):
    """
    Receive user feedback on a prediction.

    When a user confirms or corrects the predicted class,
    the image is saved to training_data/ for future retraining.

    Body:
        prediction_id: UUID of the saved prediction image
        confirmed_class: class_id ("early_blight", "late_blight", "healthy")
    """
    prediction_id = data.get("prediction_id")
    confirmed_class = data.get("confirmed_class")

    if not prediction_id or not confirmed_class:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: prediction_id, confirmed_class",
        )

    if confirmed_class not in CLASS_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class: {confirmed_class}. Valid: {', '.join(CLASS_IDS)}",
        )

    # Find the uploaded image
    found = False
    for ext in ALLOWED_EXTENSIONS:
        src_path = UPLOAD_DIR / f"{prediction_id}{ext}"
        if src_path.exists():
            # Copy to training_data
            dest_dir = TRAINING_DATA_DIR / confirmed_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{prediction_id}{ext}"

            # Load, verify and save
            try:
                from PIL import Image
                img = Image.open(src_path)
                img.save(dest_path, optimize=True)
                found = True
                print(f"[INFO] Feedback saved: {prediction_id} -> {confirmed_class}")
                break
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing image: {str(e)}",
                )

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Image not found for prediction_id: {prediction_id}",
        )

    # Count total training data collected
    total_count = sum(
        len(list((TRAINING_DATA_DIR / cid).glob("*")))
        for cid in CLASS_IDS
        if (TRAINING_DATA_DIR / cid).exists()
    )

    return {
        "success": True,
        "message": "Feedback received. Thank you!",
        "confirmed_class": confirmed_class,
        "total_feedback_samples": total_count,
    }


# -------------------- Main --------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
