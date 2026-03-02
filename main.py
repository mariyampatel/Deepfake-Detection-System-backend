from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import random

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Deepfake Detection API (Dummy Mode) Running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()

        # Check file type
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_type = "Image"
        elif filename.endswith((".mp4", ".avi", ".mov")):
            file_type = "Video"
        else:
            return JSONResponse(
                {"error": "Unsupported file type. Please upload image or video."}
            )

        # Dummy prediction
        prediction = random.choice(["Real", "Fake"])
        confidence = round(random.uniform(0.75, 0.99), 4)

        return {
            "file_type": file_type,
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse({"error": str(e)})
