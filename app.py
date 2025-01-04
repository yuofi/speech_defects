import whisper
import logging
import tempfile
import os
import librosa
import numpy as np
import re
import Levenshtein
import tensorflow as tf
import schemas
import models
import jwt
import datetime

from models import User, TokenTable
from fastapi.security import OAuth2PasswordBearer
from auth_bearer import JWTBearer
from functools import wraps
from contextlib import contextmanager
from database import Base, engine, SessionLocal
from sqlalchemy.orm import Session
from utils_register import create_access_token,create_refresh_token,verify_password,get_hashed_password
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils_api import get_features

ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days
ALGORITHM = "HS256"
JWT_SECRET_KEY = "P-x7iheWwxuQdc_53Sqc6754EGJO0TkXh7t070SPKuY"   # should be kept secret
JWT_REFRESH_SECRET_KEY = "i009kap21PU_qotFBu33kCO2xcTLFfwncpOw0NQDyGI"

Base.metadata.create_all(engine)
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

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
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"], 
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
    

@app.post("/register")
def register_user(user: schemas.UserCreate, session: Session = Depends(get_session)):
    logging.info(f"Received user data: {user}")
    existing_user = session.query(models.User).filter_by(email=user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    encrypted_password = get_hashed_password(user.password)

    new_user = models.User(username=user.username, email=user.email, password=encrypted_password )

    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return {"message":"user created successfully"}

@app.post('/login' ,response_model=schemas.TokenSchema)
def login(request: schemas.requestdetails, db: Session = Depends(get_session)):
    user = db.query(User).filter(User.email == request.email).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email")
    hashed_pass = user.password
    if not verify_password(request.password, hashed_pass):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    access=create_access_token(user.id)
    refresh = create_refresh_token(user.id)

    token_db = models.TokenTable(user_id=user.id, access_toke=access, refresh_toke=refresh, status=True)
 
    db.add(token_db)
    db.commit()
    db.refresh(token_db)
    logging.info(f"User {user.email} logged in successfully with token {access}, refresh token {refresh}")
    return {
        "access_token": access,
        "refresh_token": refresh,
    }
    
@app.get('/getusers')
def getusers( dependencies=Depends(JWTBearer()),session: Session = Depends(get_session)):
    user = session.query(models.User).all()
    return user

@app.get('/getuser', response_model=schemas.UserCreate)
def getuser(session: Session = Depends(get_session), token: str = Depends(JWTBearer())):
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload["sub"]
    user = session.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post('/change-password')
def change_password(request: schemas.changepassword, db: Session = Depends(get_session)):
    user = db.query(models.User).filter(models.User.email == request.email).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User not found")
    
    if not verify_password(request.old_password, user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid old password")
    
    encrypted_password = get_hashed_password(request.new_password)
    user.password = encrypted_password
    db.commit()
    
    return {"message": "Password changed successfully"}

@app.post('/logout')
def logout(dependencies=Depends(JWTBearer()), db: Session = Depends(get_session)):
    token=dependencies
    payload = jwt.decode(token, JWT_SECRET_KEY, ALGORITHM)
    user_id = payload['sub']
    token_record = db.query(models.TokenTable).all()
    info=[]
    for record in token_record :
        print("record",record)
        if (datetime.utcnow() - record.created_date).days >1:
            info.append(record.user_id)
    if info:
        existing_token = db.query(models.TokenTable).where(TokenTable.user_id.in_(info)).delete()
    db.commit()
        
    existing_token = db.query(models.TokenTable).filter(models.TokenTable.user_id == user_id, models.TokenTable.access_toke==token).first()
    if existing_token:
        existing_token.status=False
        db.add(existing_token)
        db.commit()
        db.refresh(existing_token)
    return {"message":"Logout Successfully"} 

def token_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
    
        payload = jwt.decode(kwargs['dependencies'], JWT_SECRET_KEY, ALGORITHM)
        user_id = payload['sub']
        data= kwargs['session'].query(models.TokenTable).filter_by(user_id=user_id,access_toke=kwargs['dependencies'],status=True).first()
        if data:
            return func(kwargs['dependencies'],kwargs['session'])
        
        else:
            return {'msg': "Token blocked"}
        
    return wrapper

@app.get("/progress", response_model=list[schemas.ProgressResponse])
def get_user_progress(session: Session = Depends(get_session), token: str = Depends(JWTBearer())):
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload["sub"]

    progress = session.query(models.Progress).filter(models.Progress.user_id == user_id).all()
    return progress

@app.post("/progress/update")
def update_progress(
    progress_data: schemas.ProgressUpdate, 
    session: Session = Depends(get_session), 
    token: str = Depends(JWTBearer())
):
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload["sub"]

    progress_entry = session.query(models.Progress).filter(
        models.Progress.user_id == user_id,
        models.Progress.course_name == progress_data.course_name
    ).first()

    if not progress_entry:
        raise HTTPException(status_code=404, detail="Progress entry not found")

    if progress_data.completed_tasks > progress_entry.total_tasks:
        raise HTTPException(status_code=400, detail="Completed tasks cannot exceed total tasks")

    progress_entry.completed_tasks = progress_data.completed_tasks
    session.commit()
    session.refresh(progress_entry)
    return {"message": "Progress updated successfully"}

@app.post("/progress/create")
def create_progress(
    progress_data: schemas.ProgressCreate, 
    session: Session = Depends(get_session), 
    token: str = Depends(JWTBearer())
):
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload["sub"]

    existing_progress = session.query(models.Progress).filter(
        models.Progress.user_id == user_id,
        models.Progress.course_name == progress_data.course_name
    ).first()

    if existing_progress:
        raise HTTPException(status_code=400, detail="Progress entry already exists")

    new_progress = models.Progress(
        user_id=user_id,
        course_name=progress_data.course_name,
        completed_tasks=progress_data.completed_tasks,
        total_tasks=progress_data.total_tasks
    )
    session.add(new_progress)
    session.commit()
    session.refresh(new_progress)
    return {"message": "Progress created successfully"}
