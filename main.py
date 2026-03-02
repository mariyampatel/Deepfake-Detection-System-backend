from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
import cv2
import tempfile

app = FastAPI()

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "image_deepfake_model.h5"
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
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
         raise ValueError("Uploaded image is not in standard RGB format.")
         
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Deepfake Detection API is Running Successfully 🚀 (Supports Images & Videos)"}

# -----------------------------
# Prediction Route (Images & Videos)
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        # --- IMAGE PROCESSING ---
        if file.content_type.startswith("image/"):
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            processed_image = preprocess_image(image)

            prediction = model(processed_image, training=False)
            confidence = float(prediction[0][0])

            result = "Fake" if confidence > 0.5 else "Real"
            return {
                "type": "image",
                "prediction": result,
                "confidence": round(confidence, 4)
            }

        # --- VIDEO PROCESSING ---
        elif file.content_type.startswith("video/"):
            # Save video to a temporary file so OpenCV can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(await file.read())
                temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract 10 evenly spaced frames to keep processing fast
            frames_to_extract = 10
            step = max(1, frame_count // frames_to_extract)

            confidences = []

            for i in range(frames_to_extract):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert OpenCV BGR format to PIL RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                processed = preprocess_image(pil_img)
                pred = model(processed, training=False)
                confidences.append(float(pred[0][0]))

            cap.release()
            os.remove(temp_video_path) # Clean up the temp file

            if not confidences:
                return JSONResponse(status_code=400, content={"error": "Could not extract frames from video."})

            # Calculate average confidence across all extracted frames
            avg_confidence = sum(confidences) / len(confidences)
            result = "Fake" if avg_confidence > 0.5 else "Real"

            return {
                "type": "video",
                "prediction": result,
                "confidence": round(avg_confidence, 4),
                "frames_analyzed": len(confidences)
            }

        else:
            return JSONResponse(status_code=400, content={"error": "Invalid file format. Please upload an image or video."})

    except ValueError as ve:
         return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {str(e)}"})
