import numpy as np
import keras
import joblib
import librosa
import boto3
import tempfile
import os
import requests
from urllib.parse import urlparse

# model = keras.models.load_model("D:\Projects\Sonex\music-classifier-service\music-classifier-service\assets\mood_model.h5")
model = keras.models.load_model(
    r"assets/mood_model.h5"
)

scaler = joblib.load(r"assets/scaler.pkl")
encoder = joblib.load(r"assets/label_encoder.pkl")


# === S3 Download Function ===
def download_from_s3(s3_url):
    """Download file from S3 URL to temporary local file"""
    try:
        # Parse S3 URL
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Download from S3
        s3_client.download_file(bucket_name, key, temp_file_path)
        return temp_file_path
    except Exception as e:
        print(f"Failed to download from S3: {e}")
        return None

def download_from_s3_https_url(url):
    """Download file from S3 HTTPS URL using boto3 with credentials"""
    try:
        print(f"Attempting to download from S3 HTTPS URL: {url}")
        
        # Parse S3 HTTPS URL to extract bucket and key
        # URL format: https://bucket.s3.amazonaws.com/key or https://s3.amazonaws.com/bucket/key
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        
        if '.s3.amazonaws.com' in parsed_url.netloc:
            # Format: https://bucket.s3.amazonaws.com/key
            bucket_name = parsed_url.netloc.split('.s3.amazonaws.com')[0]
            key = parsed_url.path.lstrip('/')
        elif 's3.amazonaws.com' in parsed_url.netloc:
            # Format: https://s3.amazonaws.com/bucket/key
            path_parts = parsed_url.path.lstrip('/').split('/', 1)
            bucket_name = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid S3 URL format: {url}")
        
        print(f"Extracted bucket: {bucket_name}, key: {key}")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file_path = temp_file.name
        temp_file.close()
        
        print(f"Created temporary file: {temp_file_path}")
        
        # Use boto3 to download
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, key, temp_file_path)
        
        print(f"Successfully downloaded file using boto3 to: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"Failed to download from S3 HTTPS URL: {e}")
        return None

def download_from_https_url(url):
    """Download file from HTTPS URL to temporary local file"""
    try:
        print(f"Attempting to download from URL: {url}")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file_path = temp_file.name
        temp_file.close()
        
        print(f"Created temporary file: {temp_file_path}")
        
        # Download from URL
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        print(f"HTTP response status: {response.status_code}")
        
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded file to: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"Failed to download from URL: {e}")
        return None

# === Define feature extraction ===
def extract_mfcc_from_mp3(file_path, max_len=130, n_mfcc=20):
    temp_file_path = None
    try:
        print(f"Processing file: {file_path}")
        
        # Check if it's an S3 URL
        if file_path.startswith('s3://'):
            print("Detected S3:// URL")
            temp_file_path = download_from_s3(file_path)
            if temp_file_path is None:
                return None
            actual_file_path = temp_file_path
        elif file_path.startswith('https://') and ('s3.amazonaws.com' in file_path):
            # Handle S3 HTTPS URLs with boto3
            print("Detected S3 HTTPS URL")
            temp_file_path = download_from_s3_https_url(file_path)
            if temp_file_path is None:
                print("Failed to download from S3 HTTPS URL")
                return None
            actual_file_path = temp_file_path
        elif file_path.startswith('https://') or file_path.startswith('http://'):
            # Handle regular HTTPS/HTTP URLs
            print("Detected regular HTTPS/HTTP URL")
            temp_file_path = download_from_https_url(file_path)
            if temp_file_path is None:
                print("Failed to download from HTTPS URL")
                return None
            actual_file_path = temp_file_path
        else:
            print("Processing local file")
            actual_file_path = file_path
        
        print(f"Loading audio from: {actual_file_path}")
        y, sr = librosa.load(actual_file_path, sr=None)
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
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"Cleaning up temporary file: {temp_file_path}")
            os.unlink(temp_file_path)

# === Predict mood ===
def predict_mood_from_mp3(file_path):
    mfcc = extract_mfcc_from_mp3(file_path)
    if mfcc is None:
        return "Error extracting features"

    # Check if model is loaded
    if model is None:
        return "Error: Model not loaded"

    # Normalize
    mfcc_scaled = scaler.transform(mfcc)
    mfcc_scaled = mfcc_scaled.reshape(1, 130, 20)  # Shape expected by model

    # Predict
    predictions = model.predict(mfcc_scaled)
    predicted_index = np.argmax(predictions)
    predicted_mood = encoder.inverse_transform([predicted_index])[0]
    # return predicted_mood
    return predicted_mood

def mood_index_to_label(index):
    mood_labels = {
        0: "happy",
        1: "sad",
        2: "relax",
        3: "angry"
    }
    return mood_labels.get(index, "Unknown")

# file_path = r"assets/sampletracks/sample.mp3"
# mood = predict_mood_from_mp3(file_path)
# print("Predicted mood:", mood)
# mood = mood_index_to_label(mood)
