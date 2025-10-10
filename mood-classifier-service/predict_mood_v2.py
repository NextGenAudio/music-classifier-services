import numpy as np
import librosa 
import joblib
from urllib.parse import urlparse
import boto3
import tempfile
import requests
import os
import keras

# Load model and encoders
try:
    model = keras.models.load_model(r"assets/mood_model_v2.h5")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    encoder = joblib.load(r"assets/label_encoder_v2.pkl")
    print("Label encoder loaded successfully")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    encoder = None

# Note: scaler.pkl doesn't exist, you may need to create it or use a different approach
scaler = None
try:
    scaler = joblib.load(r"assets/scaler.pkl")
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Warning: Scaler not found: {e}")
    print("You may need to create a scaler for feature normalization")


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





# function for Music Information Retrieval
def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, hop_length=20).T, axis=0)
    result = np.hstack((result, zcr))
    #print("zcr shape",zcr.shape)

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=20, hop_length=20).T, axis=0)
    result = np.hstack((result, chroma_stft))
    #print("chroma_stft shape",chroma_stft.shape)

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
    result = np.hstack((result, mfcc))
    #print("mfcc shape",mfcc.shape)

    rms = np.mean(librosa.feature.rms(y=data, frame_length=100).T, axis=0)
    result = np.hstack((result, rms))
    #print("rms shape",rms.shape)

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, hop_length=20).T, axis=0)
    result = np.hstack((result, mel))
    #print("mel shape",mel.shape)

    return result

def feature_extractor(path):
    data, sample_rate = librosa.load(path)
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    return result

def predict_mood_from_features(feature):
    """Predict mood from extracted features"""
    if model is None:
        return "Error: Model not loaded"
    if encoder is None:
        return "Error: Label encoder not loaded"
    
    prediction = np.argmax(model.predict(feature))
    predict_encode = np.array([])
    for i in range(5):
        if i == prediction:
            predict_encode = np.append(predict_encode, [1])
        else:
            predict_encode = np.append(predict_encode, [0])
    predict_encode = predict_encode.reshape(1, -1)
    predicted_mood = encoder.inverse_transform(predict_encode)
    return predicted_mood[0][0]

def predict_mood_from_mp3(file_path):
    """Main function to predict mood from MP3 file"""
    try:
        # Extract features using the comprehensive feature extraction
        temp_file_path = None
        
        # Handle different file path types (S3, HTTPS, local)
        if file_path.startswith('s3://'):
            print("Detected S3:// URL")
            temp_file_path = download_from_s3(file_path)
            if temp_file_path is None:
                return "Error downloading from S3"
            actual_file_path = temp_file_path
        elif file_path.startswith('https://') and ('s3.amazonaws.com' in file_path):
            print("Detected S3 HTTPS URL")
            temp_file_path = download_from_s3_https_url(file_path)
            if temp_file_path is None:
                return "Error downloading from S3 HTTPS URL"
            actual_file_path = temp_file_path
        elif file_path.startswith('https://') or file_path.startswith('http://'):
            print("Detected regular HTTPS/HTTP URL")
            temp_file_path = download_from_https_url(file_path)
            if temp_file_path is None:
                return "Error downloading from HTTPS URL"
            actual_file_path = temp_file_path
        else:
            print("Processing local file")
            actual_file_path = file_path
        
        # Extract comprehensive features
        features = feature_extractor(actual_file_path)
        if features is None:
            return "Error extracting features"
        
        # Scale features (if scaler is available)
        if scaler is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            print("Warning: No scaler available, using raw features")
            features_scaled = features.reshape(1, -1)
        
        # Predict mood
        mood = predict_mood_from_features(features_scaled)
        
        return mood
        
    except Exception as e:
        print(f"Error predicting mood: {e}")
        return "Error predicting mood"
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"Cleaning up temporary file: {temp_file_path}")
            os.unlink(temp_file_path)

def mood_index_to_label(index):
    """Convert mood index to human-readable label"""
    mood_labels = {
        0: "happy",
        1: "sad", 
        2: "relax",
        3: "angry",
        4: "energetic"  # assuming 5 classes based on the loop
    }
    return mood_labels.get(index, "Unknown")

# Test function
def test_prediction():
    """Test the prediction with a sample file"""
    sample_file = r"sampletracks/sample.mp3"
    if os.path.exists(sample_file):
        print(f"Testing prediction with: {sample_file}")
        mood = predict_mood_from_mp3(sample_file)
        print(f"Predicted mood: {mood}")
    else:
        print(f"Sample file not found: {sample_file}")

if __name__ == "__main__":
    test_prediction()

# Remove the test code at the bottom
# feature1 = feature_extractor(r"D:\Projects\Sonex\music-classifier-service\music-classifier-service\sampletracks\sample.mp3").reshape(1,162)  
# print("MyGO!!!!! Hekitenbansou:")
# print("classification:",predict_decode(feature1))
# print("-------------------------------------")
