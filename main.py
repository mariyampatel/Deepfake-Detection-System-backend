from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

app = FastAPI()

# -----------------------------
# Load Model (With Path Checking)
# -----------------------------
# Updated path to look inside the 'model' folder
MODEL_PATH = "model/image_deepfake_model.h5"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model file: {e}")
else:
    print(f"❌ ALERT: Model file NOT FOUND at {MODEL_PATH}. Please check the folder!")

# -----------------------------
# Image Preprocessing Function
# -----------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))   # Ensure this matches your model's training size exactly!
    image_array = np.array(image) / 255.0
    
    # Ensure it's a 3-channel RGB image before sending to model
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
         raise ValueError("Uploaded image is not in standard RGB format.")
         
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

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
        return JSONResponse(status_code=500, content={"error": "Model not loaded. Check server terminal for reasons."})

    # Basic check to ensure an image is uploaded
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Invalid file format. Please upload an image."})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        # Using model(image) is faster and safer for single predictions than model.predict()
        prediction = model(processed_image, training=False)
        
        # Convert TensorFlow tensor to float
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = "Fake"
        else:
            result = "Real"

        return {
            "prediction": result,
            "confidence": round(confidence, 4)
        }

    except ValueError as ve:
         return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        # This will return the EXACT error to your Postman/Frontend
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {str(e)}"})
