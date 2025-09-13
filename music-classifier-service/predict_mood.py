import numpy as np
from tensorflow import keras
import joblib
import librosa

# model = keras.models.load_model("D:\Projects\Sonex\music-classifier-service\music-classifier-service\assets\mood_model.h5")
model = keras.models.load_model(
    r"D:\Projects\Sonex\music-classifier-service\music-classifier-service\assets\mood_model.h5"
)

scaler = joblib.load(r"D:\Projects\Sonex\music-classifier-service\music-classifier-service\assets\scaler.pkl")
encoder = joblib.load(r"D:\Projects\Sonex\music-classifier-service\music-classifier-service\assets\label_encoder.pkl")


# === Define feature extraction ===
def extract_mfcc_from_mp3(file_path, max_len=130, n_mfcc=20):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # Shape: (time_steps, n_mfcc)

        # Pad or truncate
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len, :]
        return mfcc
    except Exception as e:
        print("Failed to process file:", e)
        return None

# === Predict mood ===
def predict_mood_from_mp3(file_path):
    mfcc = extract_mfcc_from_mp3(file_path)
    if mfcc is None:
        return "Error extracting features"

    # Normalize
    mfcc_scaled = scaler.transform(mfcc)
    mfcc_scaled = mfcc_scaled.reshape(1, 130, 20)  # Shape expected by model

    # Predict
    predictions = model.predict(mfcc_scaled)
    predicted_index = np.argmax(predictions)
    predicted_mood = encoder.inverse_transform([predicted_index])[0]
    return predicted_mood

def mood_index_to_label(index):
    mood_labels = {
        0: "happy",
        1: "sad",
        2: "relax",
        3: "angry"
    }
    return mood_labels.get(index, "Unknown")

file_path = r"D:\\Projects\\Sonex\\music-classifier-service\\music-classifier-service\\sampletracks\\sample.mp3"

# mood = predict_mood_from_mp3(file_path)
# mood = mood_index_to_label(mood)
