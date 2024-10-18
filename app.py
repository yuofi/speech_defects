from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import whisper
import logging
from contextlib import contextmanager
import tempfile
import os
import keras
import librosa
import numpy as np
import re
import Levenshtein
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware
from utils import (
    extract_features,
    pad_or_trim,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler()] 
)

os.environ['NUMBA_CACHE_DIR'] = '/tmp'

app = FastAPI(port=8000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

filepath = os.path.abspath("cnn_1_v6_final_model.h5")
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Model file not found at {filepath}")

model = keras.models.load_model(filepath, compile=False)
whisper_model = whisper.load_model("tiny")

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

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Defects_model API"}

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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Log to console
)



@app.post("/process-audio")
async def process_audio(
    audio: UploadFile = File(...), 
    phrase: str = Form(...)
):
    if audio.content_type != "audio/mpeg":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only MP3 files are supported."
        )

    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Received empty file")

        logging.info(f"Received audio bytes: {len(audio_bytes)} bytes")

        with temporary_audio_file(audio_bytes) as tmp_filename:
            logging.info(f"Temporary file created: {tmp_filename}")

            audio_data, sample_rate = librosa.load(tmp_filename, sr=None)
            logging.info(
                f"Audio loaded: sample rate = {sample_rate}, data shape = {audio_data.shape}"
            )
            if not audio_data.any() or sample_rate == 0:
                raise ValueError("Empty or invalid audio data.")
            
            # Извлекаем признаки из аудиоданных
            features = extract_features(audio_data, sample_rate)
            logging.info(f"Features extracted: shape = {features.shape}")

            target_shape = (1, model.input_shape[1])
            features = pad_or_trim(features, target_shape[1])
            features = np.expand_dims(features, axis=0)

            prediction = model.predict(features)
            logging.info(f"Prediction: {prediction}")

            transcription_result = whisper_model.transcribe(tmp_filename, language="russian")
            transcribed_text = transcription_result["text"].lower().strip()

            # Удаление знаков препинания из транскрибированного текста
            transcribed_text_clean = re.sub(r'[^\w\s]', '', transcribed_text) 
            logging.info(f"Transcribed text (cleaned): {transcribed_text_clean}")

            # Вычисление редакторского расстояния
            lev_distance = Levenshtein.distance(transcribed_text_clean, phrase.lower().strip())
            phrase_length = max(len(transcribed_text_clean), len(phrase))

            # Допускаем различие в 40% длины исходной фразы
            max_acceptable_distance = 0.5 * phrase_length
            match_phrase = lev_distance <= max_acceptable_distance

            logging.info(f"Expected phrase: {phrase}, Is correct: {match_phrase}, Transcribed text: {transcribed_text_clean}, Levenshtein distance: {lev_distance}")

            return {
                "prediction": prediction.tolist(),
                "match_phrase": match_phrase,
                "lev_distance": lev_distance,
                "transcribed_text": transcribed_text_clean
            }

    except Exception as e:
        logging.exception(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
