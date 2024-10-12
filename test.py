import os
import numpy as np
import keras
import httpx
import librosa

from utils import (
    extract_features,
    pad_or_trim,
)

def test_get_answer(audio_file_path: str):
    url = "http://127.0.0.1:8000/process-audio"
    headers = {
        "accept": "application/json",
    }

    with open(audio_file_path, "rb") as audio_file:
        files = {
            "audio": ("test.mp3", audio_file, "audio/mp3")
        }
        response = httpx.post(url, headers=headers, files=files)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())


audio_file_path = "test_audio.mp3"
if not os.path.exists(audio_file_path):
    raise FileNotFoundError(f"Audio file not found at {audio_file_path}")

audio_data, sample_rate = librosa.load(audio_file_path)

features = extract_features(audio_data, sample_rate) 

target_shape = (32, 200) 
features = pad_or_trim(features, target_shape[1])


features = np.expand_dims(features, axis=0)

filepath = os.path.abspath("cnn_1_v6_final_model.h5")
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Model file not found at {filepath}")

model = keras.models.load_model(filepath, compile=False)


prediction = model.predict(features)
print(f"Prediction: {prediction.tolist()}")

test_get_answer(audio_file_path)