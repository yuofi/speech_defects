import librosa
import numpy as np
from keras import layers, models

def create_cnn_model(input_shape):
    model = models.Sequential()

    # First Convolutional Layer
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Second Convolutional Layer
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(512, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(512, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def get_features(path, duration=6):
    try:
        # Load audio file with specific duration and offset to handle silent parts
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None  # Skip the file if there's an error

    # Without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # With noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))

    # Stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))

    return result


def extract_features(data, sample_rate, target_shape=40):
    result = np.array([])

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=data)
    zcr = np.mean(zcr.T, axis=0)
    zcr = pad_or_trim(zcr, target_shape)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_stft = np.mean(chroma_stft.T, axis=0)
    chroma_stft = pad_or_trim(chroma_stft, target_shape)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = pad_or_trim(mfcc, target_shape)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = librosa.feature.rms(y=data)
    rms = np.mean(rms.T, axis=0)
    rms = pad_or_trim(rms, target_shape)
    result = np.hstack((result, rms))

    # MelSpectrogram
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    mel = np.mean(mel.T, axis=0)
    mel = pad_or_trim(mel, target_shape)
    result = np.hstack((result, mel))

    return result


def pad_or_trim(feature, target_shape):
    """Pad or trim feature array to ensure a consistent shape."""
    if len(feature) > target_shape:
        feature = feature[:target_shape]
    elif len(feature) < target_shape:
        feature = np.pad(feature, (0, target_shape - len(feature)), mode='constant')
    return feature


def noise(data, noise_factor=0.005):
    noise_amp = noise_factor * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sample_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_factor)
