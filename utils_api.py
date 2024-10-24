import librosa
import numpy as np
from keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout, Reshape, Input, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import soundfile as sf
import tensorflow as tf

# Load the embedding model globally to avoid reloading it every time
embedding_model = hub.load("https://www.kaggle.com/models/google/speech-embedding/TensorFlow1/speech-embedding/1")

def load_audio(audio_file_path):
    try:
        audio_samples, sample_rate = librosa.load(audio_file_path, sr=None)
        audio_samples = audio_samples.astype(np.float32)
        audio_samples /= np.max(np.abs(audio_samples))
        return audio_samples, sample_rate
    except Exception as e:
        print(f"Error loading {audio_file_path}: {e}")
        return None, None

def pad_or_trim(audio, sr, target_length=5):
    target_samples = int(target_length * sr)
    return librosa.util.fix_length(audio, size=target_samples) if len(audio) < target_samples else audio[:target_samples]

def get_features(path, duration=5):
    try:
        data, sample_rate = load_audio(path)

        # Model need 16000 sample rate
        if sample_rate != 16000:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        data = pad_or_trim(data, sample_rate)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    
    embeddings = extract_embeddings(np.expand_dims(data, axis=0))  # Add batch dimension

    if embeddings is not None:
        return embeddings
    else:
        return None # Fail

def extract_embeddings(audio_samples):
    """Extract embeddings from audio samples."""
    try:
        # Convert audio samples to tensor and extract embeddings
        embeddings = embedding_model.signatures['default'](tf.convert_to_tensor(audio_samples))
        return embeddings['default'].numpy().flatten()
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return None  # Return None if there's an error

def pad_or_trim(feature, target_shape):
    """Pad or trim feature array to ensure a consistent shape."""
    if len(feature) > target_shape:
        feature = feature[:target_shape]
    elif len(feature) < target_shape:
        feature = np.pad(feature, (0, target_shape - len(feature)), mode='constant')
    return feature