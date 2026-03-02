from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

# -------------------------------
# FIX FOR RENDER MODEL PATH
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "image_deepfake_model.h5")

print("Loading model from:", model_path)

# Load model
model = load_model(model_path)

# -------------------------------
# IMAGE PREPROCESS FUNCTION
# -------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # change if your model size different
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# HOME ROUTE
# -------------------------------
@app.get("/")
def home():
    return {"message": "Deepfake Detection API is Running"}

# -------------------------------
# PREDICTION ROUTE
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = "Fake"
        else:
            result = "Real"

        return JSONResponse({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e)
        })
