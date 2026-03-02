from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD TRAINED MODEL ----------------
model = load_model("model/image_deepfake_model.h5")

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- REAL MODEL PREDICTION ----------------
def predict_frame(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        return "Fake", float(prediction)
    else:
        return "Real", float(1 - prediction)


# ================= IMAGE ENDPOINT =================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image file"}

    prediction, confidence = predict_frame(image)

    return {
        "prediction": prediction,
        "confidence": confidence
    }


# ================= VIDEO ENDPOINT =================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(await file.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)

    predictions = []
    confidences = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            prediction, confidence = predict_frame(frame)
            predictions.append(prediction)
            confidences.append(confidence)

        frame_count += 1

        if frame_count >= 200:
            break

    cap.release()

    if not predictions:
        return {"error": "Could not process video"}

    final_prediction = max(set(predictions), key=predictions.count)
    avg_confidence = sum(confidences) / len(confidences)

    return {
        "prediction": final_prediction,
        "confidence": float(avg_confidence)
    }


@app.get("/")
def home():
    return {"message": "Deepfake Detection Backend Running with AI Model"}
