from fastapi import FastAPI, File, UploadFile, HTTPException
from models import User, Course, connection
from forms import UserRegistration, UserLoginForm
from fastapi.responses import JSONResponse
from utils import create_cnn_model, get_features, extract_features, pad_or_trim, noise, stretch, pitch
from peewee import *
import numpy as np
import tensorflow as tf
import keras
import requests
import io
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = 'audio'
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_SERVER_URL = "http://model-server-url/predict"

@app.post("/save-audio")
async def save_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return JSONResponse(content={"message": "File saved successfully", "filePath": file_path}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    
model = tf.keras.models.load_model("cnn_1_v6_final_model.keras", compile=False)

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    if audio.content_type != "audio/mpeg":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an MP3 file.")

    audio_bytes = await audio.read()

    features = get_features(audio_bytes)

    if features is None:
        raise HTTPException(status_code=400, detail="Invalid audio file. Please upload a valid MP3 file.")

    prediction = model.predict(np.expand_dims(features, axis=0))

    return {"prediction": prediction}


'''
@router.post("/login")
async def login(user_data: UserLoginForm):
    user = User.get(User.login == user_data.login)
    if not user or user_data.password != user.password:
        return {"message": "Invalid login or password"}
    token_content = {"user_id": user.user_id}
    jwt_token = jwt.encode(token_content, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": jwt_token}


@router.post("/registration")
async def registration(user_data: UserRegistration):
    try:
        new_user = User.create(login=user_data.login, password=user_data.password)
        new_user.save()
        return {"message": "User registered successfully"}
    except IntegrityError:
        return {"message": "User with this login already exists"} 
'''
