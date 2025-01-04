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


def process_embeddings(embeddings):
    """
    Flatten embeddings and pad or trim to match the expected feature size of 5088.
    Embeddings typically come in shape (1, 36, 1, 96), so they are first flattened and reshaped.
    """
    # Flatten the embeddings from (1, 36, 1, 96) to (1, 3456)
    flattened = embeddings.reshape(1, -1)
    
    # Pad or trim to match the expected feature size of 5088
    if flattened.shape[1] > 5088:
        padded = flattened[:, :5088]  # Trim if too large
    else:
        padded = np.pad(flattened, ((0, 0), (0, 5088 - flattened.shape[1])), mode='constant')
    
    # The output should be of shape (1, 5088)
    return padded  # No need to reshape again since it's already (1, 5088)


def reshape_features(features):
    """
    Reshape features to match the model's expected input shape of (1, 5088, 1)
    """
    # Reshape to [1, 5088, 1] to add the extra dimension if needed
    features = np.reshape(features, (1, 5088, 1))
    return features


def get_features(path, duration=5):
    """
    Load the audio file, preprocess it, extract embeddings, and process them for the model.
    """
    try:
        # Load and preprocess audio
        logging.info(f"Loading audio file: {path}")
        data, sample_rate = librosa.load(path, duration=duration, offset=0.6)
        
        if data is None or len(data) == 0:
            logging.error("Audio data is empty")
            return None
            
        logging.info(f"Original audio shape: {data.shape}, sample rate: {sample_rate}")

        # Ensure audio length is exactly 5 seconds (16000 * 5 samples)
        target_length = 16000 * 5
        data = pad_or_trim(data, target_length)
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            logging.info(f"Resampling from {sample_rate}Hz to 16000Hz")
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Ensure the audio is mono and has the correct shape
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Add batch dimension and convert to tensor
        audio_tensor = tf.convert_to_tensor(data[np.newaxis, :], dtype=tf.float32)
        logging.info(f"Processed audio tensor shape: {audio_tensor.shape}")

        # Extract embeddings using the embedding model
        try:
            result = embedding_model.signatures['default'](audio_tensor)
            embeddings = result['default'].numpy()
            
            if embeddings is None:
                logging.error("Failed to extract embeddings")
                return None
                
            logging.info(f"Extracted embeddings shape: {embeddings.shape}")
            
            # Process embeddings to match model input shape
            processed_embeddings = process_embeddings(embeddings)
            logging.info(f"Processed embeddings shape: {processed_embeddings.shape}")
            
            return processed_embeddings

        except Exception as e:
            logging.error(f"Error in embedding extraction: {e}")
            return None

    except Exception as e:
        logging.error(f"Error in audio processing: {e}")
        return None


def pad_or_trim(feature, target_length):
    """
    Pad or trim the input feature to ensure it matches the target length.
    If the feature is longer than target_length, trim it.
    If shorter, pad with zeros.
    """
    if len(feature) > target_length:
        return feature[:target_length]  # Trim if too long
    else:
        return np.pad(feature, (0, target_length - len(feature)), mode='constant')  # Pad if too short
