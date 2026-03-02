from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = FastAPI()

# -----------------------------
# Load Model (Root Folder)
# -----------------------------
model_path = "image_deepfake_model.h5"

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully")
    else:
        print(f"Model file not found at: {model_path}")
        model = None
except Exception as e:
    print("Error loading model:", e)
    model = None

# -----------------------------
# Image Preprocessing Function
# -----------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Change size if your model different
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Deepfake Detection API is Running Successfully 🚀"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded properly"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        confidence = float(prediction[0][0])

        result = "Fake" if confidence > 0.5 else "Real"

        return {
            "prediction": result,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)})
