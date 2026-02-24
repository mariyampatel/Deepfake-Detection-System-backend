from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tempfile

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Dummy Model Logic ----------------
def predict_frame(image):
    image = cv2.resize(image, (224, 224))
    mean_pixel = np.mean(image)

    if mean_pixel > 127:
        return "Real", 0.85
    else:
        return "Fake", 0.90


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
        "confidence": float(confidence)
    }


# ================= VIDEO ENDPOINT =================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    # Save uploaded video temporarily
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

        # Process every 10th frame
        if frame_count % 10 == 0:
            prediction, confidence = predict_frame(frame)
            predictions.append(prediction)
            confidences.append(confidence)

        frame_count += 1

        # Limit frames for performance
        if frame_count >= 200:
            break

    cap.release()

    if not predictions:
        return {"error": "Could not process video"}

    # Majority Voting
    final_prediction = max(set(predictions), key=predictions.count)

    # Average Confidence
    avg_confidence = sum(confidences) / len(confidences)

    return {
        "prediction": final_prediction,
        "confidence": float(avg_confidence)
    }


@app.get("/")
def home():
    return {"message": "Deepfake Detection Backend Running"}
