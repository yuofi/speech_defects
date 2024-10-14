import logging
from contextlib import contextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import librosa
import numpy as np
import keras
from utils import (
    create_cnn_model,
    get_features,
    extract_features,
    pad_or_trim,
    noise,
    stretch,
    pitch,
)

app = FastAPI(port=8000)

# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     # Add more origins if needed
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

filepath = os.path.abspath("cnn_1_v6_final_model.h5")
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Model file not found at {filepath}")

model = keras.models.load_model(filepath, compile=False)
target_shape = (32, 200)


@app.post("/save-audio")
async def save_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join("audio", file.filename)
    os.makedirs("audio", exist_ok=True)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return JSONResponse(
            content={"message": "File saved successfully", "filePath": file_path},
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

log_file_path = os.path.join("/tmp", "server.log")

logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@contextmanager
def temporary_audio_file(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        tmp_filename = tmp_file.name
    try:
        yield tmp_filename
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    if audio.content_type != "audio/mpeg":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only MP3 files are supported."
        )

    try:
        audio_bytes = await audio.read()
        logging.info(
            f"Received audio bytes: {len(audio_bytes)} bytes"
        ) 
        with temporary_audio_file(audio_bytes) as tmp_filename:
            logging.info(f"Temporary file created: {tmp_filename}")
            audio_data, sample_rate = librosa.load(tmp_filename, sr=None)
            logging.info(
                f"Audio loaded: sample rate = {sample_rate}, data shape = {audio_data.shape}"
            )
            if not audio_data.any() or sample_rate == 0:
                raise ValueError("Empty or invalid audio data.")

            features = extract_features(audio_data, sample_rate)
            logging.info(f"Features extracted: shape = {features.shape}")
            target_shape = (1, model.input_shape[1])
            features = pad_or_trim(features, target_shape[1])
            features = np.expand_dims(features, axis=0)

            prediction = model.predict(features)

            logging.info(f"Prediction: {prediction}")
            return {"prediction": prediction.tolist()}

    except librosa.util.exceptions.ParameterError as e:
        logging.error(f"Librosa error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")
    except Exception as e:
        logging.exception(f"Error processing audio: {e}") 
        raise HTTPException(status_code=500, detail="Internal server error")

