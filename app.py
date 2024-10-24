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
import tensorflow as tf

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils_api import get_features


#вывод в консоль для просмотри на hugging face
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler()] 
)

# Установка временной директории для кэша Numba
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

# Инициализация FastAPI приложения
app = FastAPI(port=8000)

# Настройка CORS (Cross-Origin Resource Sharing) для обработки запросов с разных доменов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET, POST"],
    allow_headers=["*"],
)

# Инициализация и загрузка модели Whisper для распознавания речи
cache_dir = "/tmp/whisper_cache"
os.makedirs(cache_dir, exist_ok=True)
whisper_model = whisper.load_model("tiny", download_root=cache_dir)

# загрузка параметров модели
filepath = "best_model.keras"
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Model file not found at {filepath}")\
        
model = tf.keras.models.load_model(filepath, compile=False)
logging.info(model.summary())
# Контекстный менеджер для временных аудио файлов
@contextmanager
def temporary_audio_file(audio_bytes):
    """
    Создает временный файл для хранения аудио данных и автоматически удаляет его после использования
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        tmp_filename = tmp_file.name
    try:
        yield tmp_filename
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

# Корневой endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Defects_model API"}

# Endpoint для сохранения аудио файлов
@app.post("/save-audio")
async def save_audio(file: UploadFile = File(...)):
    """
    Обработчик для сохранения загруженных аудио файлов
    """
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

# Настройка пути для файла логов
log_file_path = os.path.join("/tmp", "server.log")

# Настройка логирования для отслеживания работы сервера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Основной endpoint для обработки аудио
@app.post("/process-audio")
async def process_audio(
    audio: UploadFile = File(...), 
    phrase: str = Form(...)
):
    """
    Главный обработчик для анализа аудио файлов:
    - Делает предсказание моделью
    - Прогоняет аудио через openai-whisper для проверки фразы
    - Сравнивает полученный текст с ожидаемой фразой
    """
    # Проверка формата файла
    if audio.content_type != "audio/mpeg":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only MP3 files are supported."
        )

    try:
        # Чтение аудио файла
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Received empty file")

        logging.info(f"Received audio bytes: {len(audio_bytes)} bytes")

        # Обработка аудио во временном файле
        with temporary_audio_file(audio_bytes) as tmp_filename:
            logging.info(f"Temporary file created: {tmp_filename}")

            # Загрузка аудио данных
            audio_data, sample_rate = librosa.load(tmp_filename, sr=None)
            logging.info(
                f"Audio loaded: sample rate = {sample_rate}, data shape = {audio_data.shape}"
            )
            if not audio_data.any() or sample_rate == 0:
                raise ValueError("Empty or invalid audio data.")
            
            # Извлечение признаков из аудио
            features = get_features(tmp_filename)
            # features = np.expand_dims(features, axis=0)  # Add batch dimension
            logging.info(f"Features extracted: shape = {features.shape}")

            # Получение предсказания от модели
            class_weights = {0: 0.5460790960451978, 1: 1.0068333333333332, 2: 1000.696369636963697}

            prediction = model.predict(features)
            logging.info(f"Prediction shape: {prediction.shape}")

            #умножаем предикт на веса классов
            for j in range(prediction.shape[1]):
                prediction[0, j] *= class_weights.get(j, 1.0)
                prediction[0, j] *= 10

            logging.info(f"Prediction: {prediction}")
            response_answer = np.argmax(prediction)
            if (response_answer == 0): 
                response_answer = 1
            else:
                response_answer = 0
            logging.info(f"Right or with defects: 1 or 0: {response_answer}")

            # Транскрибация аудио с помощью Whisper
            transcription_result = whisper_model.transcribe(tmp_filename, language="russian")
            transcribed_text = transcription_result["text"].lower().strip()

            # Очистка транскрибированного текста
            transcribed_text_clean = re.sub(r'[^\w\s]', '', transcribed_text) 
            logging.info(f"Transcribed text (cleaned): {transcribed_text_clean}")

            # Сравнение с ожидаемой фразой
            lev_distance = Levenshtein.distance(transcribed_text_clean, phrase.lower().strip())
            phrase_length = max(len(transcribed_text_clean), len(phrase))

            # Определение допустимого расстояния Левенштейна
            max_acceptable_distance = 0.5 * phrase_length
            match_phrase = lev_distance <= max_acceptable_distance

            logging.info(f"Expected phrase: {phrase}, Is correct: {match_phrase}, Transcribed text: {transcribed_text_clean}, Levenshtein distance: {lev_distance}")

            # Возврат результатов
            return {
                "prediction": response_answer,
                "match_phrase": match_phrase
            }

    except Exception as e:
        logging.exception(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")