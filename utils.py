import librosa
import numpy as np
from keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout, Reshape, Input, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import tensorflow as tf
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler()] 
)

# Load the embedding model globally to avoid reloading it every time
embedding_model = hub.load("https://www.kaggle.com/models/google/speech-embedding/TensorFlow1/speech-embedding/1")
def get_features(path, duration=5):
    try:
        # Load audio file with specific duration and offset to handle silent parts
        data, sample_rate = librosa.load(path, duration=duration, offset=0.6)
        data = pad_or_trim(data, sample_rate * 5)

        # Model need 16000 sample rate
        if sample_rate != 16000:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
    except Exception as e:
        logging.info(f"Error loading {path}: {e}")
        return None

    embeddings = extract_embeddings(np.expand_dims(data, axis=0))  # Add batch dimension

    if embeddings is not None:
        try:
            result = embedding_model.signatures['default'](tf.convert_to_tensor(data))
            result = embeddings['default'].numpy().flatten()
        except Exception as e:
            logging.info(f"Error extracting embedding from {path}: {e}")
            return None
    else:
        return None # Fail

    return result

def extract_embeddings(audio_samples):
    """Extract embeddings from audio samples."""
    try:
        # Convert audio samples to tensor and extract embeddings
        embeddings = embedding_model.signatures['default'](tf.convert_to_tensor(audio_samples))
        return embeddings['default'].numpy().flatten()
    except Exception as e:
        logging.info(f"Error extracting embeddings: {e}")
        return None  # Return None if there's an error

def pad_or_trim(feature, target_shape):
    """Pad or trim feature array to ensure a consistent shape."""
    if len(feature) > target_shape:
        feature = feature[:target_shape]
    elif len(feature) < target_shape:
        feature = np.pad(feature, (0, target_shape - len(feature)), mode='constant')
    return feature