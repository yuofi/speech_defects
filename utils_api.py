import librosa
import numpy as np
from keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout, Reshape, Input, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample, butter, lfilter
import logging

# Настройка логирования для отслеживания ошибок и процесса выполнения
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler()] 
)

# Загрузка модели для извлечения эмбеддингов из речи
embedding_model = hub.load("https://www.kaggle.com/models/google/speech-embedding/TensorFlow1/speech-embedding/1")

# Константы для фильтрации звука
LOWCUT = 400    # Нижняя частота среза в Гц
HIGHCUT = 5000  # Верхняя частота среза в Гц
ORDER = 5       # Порядок фильтра

def load_audio(audio_file_path):
    """
    Загрузка аудиофайла и нормализация его амплитуды
    Args:
        audio_file_path: путь к аудиофайлу
    Returns:
        кортеж (аудио данные, частота дискретизации)
    """
    try:
        audio_samples, sample_rate = librosa.load(audio_file_path, sr=None)
        audio_samples = audio_samples.astype(np.float32)
        audio_samples /= np.max(np.abs(audio_samples))  # Нормализация амплитуды
        return audio_samples, sample_rate
    except Exception as e:
        logging.exception(f"Error loading {audio_file_path}: {e}")
        return None, None

def pad_or_trim(audio, sr, target_length=5):
    """
    Обрезка или дополнение аудио до заданной длительности
    Args:
        audio: аудио данные
        sr: частота дискретизации
        target_length: целевая длительность в секундах
    Returns:
        обработанные аудио данные
    """
    target_samples = int(target_length * sr)
    return librosa.util.fix_length(audio, size=target_samples) if len(audio) < target_samples else audio[:target_samples]

def get_features(path, duration=5):
    """
    Извлечение признаков из аудиофайла
    Args:
        path: путь к аудиофайлу
        duration: длительность в секундах
    Returns:
        эмбеддинги аудио или None в случае ошибки
    """
    try:
        data, sample_rate = load_audio(path)
        data, sample_rate = upgrade_sound(data, sample_rate)  # Улучшение качества звука
        data = pad_or_trim(data, sample_rate)
    except Exception as e:
        logging.exception(f"Error loading {path}: {e}")
        return None
    
    data = np.array(data, dtype=np.float32)  # Преобразование в float32
    embeddings = extract_embeddings(np.expand_dims(data, axis=0))  # Добавление размерности батча

    return embeddings if embeddings is not None else None

def extract_embeddings(audio_samples):
    """
    Извлечение эмбеддингов из аудио с помощью предобученной модели
    Args:
        audio_samples: аудио данные
    Returns:
        эмбеддинги в форме (1, n_features)
    """
    try:
        # Преобразование в тензор и получение эмбеддингов
        embeddings = embedding_model.signatures['default'](tf.convert_to_tensor(audio_samples))
        
        # Получение тензора эмбеддингов
        embeddings_tensor = embeddings['default'].numpy()
        
        # Преобразование формы для соответствия входу модели
        embeddings_flat = embeddings_tensor.reshape((1, -1))
        
        return embeddings_flat
    except Exception as e:
        logging.exception(f"Error extracting embeddings: {e}")
        return None

def butter_bandpass(lowcut, highcut, sr, order=5):
    """
    Создание полосового фильтра Баттерворта
    Args:
        lowcut: нижняя частота среза
        highcut: верхняя частота среза
        sr: частота дискретизации
        order: порядок фильтра
    Returns:
        коэффициенты фильтра (b, a)
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(y, sr, lowcut=LOWCUT, highcut=HIGHCUT, order=ORDER):
    """
    Применение полосового фильтра к аудио
    Args:
        y: аудио данные
        sr: частота дискретизации
        lowcut: нижняя частота среза
        highcut: верхняя частота среза
        order: порядок фильтра
    Returns:
        отфильтрованные аудио данные
    """
    b, a = butter_bandpass(lowcut, highcut, sr, order)
    return lfilter(b, a, y)

def resample_audio(y, sr, target_sr=16000):
    """
    Передискретизация аудио до целевой частоты
    Args:
        y: аудио данные
        sr: исходная частота дискретизации
        target_sr: целевая частота дискретизации
    Returns:
        кортеж (передискретизированные данные, новая частота)
    """
    if sr != target_sr:
        num_samples = round(len(y) * float(target_sr) / sr)
        return resample(y, num_samples), target_sr
    return y, sr

def upgrade_sound(y, sr):
    """
    Комплексное улучшение качества звука
    Args:
        y: аудио данные
        sr: частота дискретизации
    Returns:
        кортеж (улучшенные аудио данные, частота дискретизации)
    
    Выполняет:
    1. Передискретизацию до 16кГц
    2. Нормализацию амплитуды
    3. Полосовую фильтрацию
    4. Предварительное усиление высоких частот
    """
    y_resampled, sr = resample_audio(y, sr)
    y_normalized = librosa.util.normalize(y_resampled)
    y_filtered = apply_bandpass_filter(y_normalized, sr)
    return librosa.effects.preemphasis(y_filtered), sr