import numpy as np
import keras
import joblib
import librosa
import boto3
import tempfile
import os
import requests
from urllib.parse import urlparse
from scipy.ndimage import zoom

# Load model files with error handling for development/demo purposes
try:
    model = keras.models.load_model(r"assets/genre_model.h5")
    scaler = joblib.load(r"assets/scaler.pkl")
    encoder = joblib.load(r"assets/label_encoder.pkl")
    print("Genre model files loaded successfully!")
except Exception as e:
    print(f"Warning: Genre model files not found ({e}). Service will return demo results.")
    model = None
    scaler = None
    encoder = None


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
def extract_features_from_mp3(file_path, target_shape=(210, 210)):
    """
    Extract features from MP3 file to create 2D spectrogram for CNN model.
    The model expects input shape (210, 210, 1) - likely a spectrogram image.
    """
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
        # Load audio with fixed sample rate
        y, sr = librosa.load(actual_file_path, sr=22050)
        
        # Create mel-spectrogram (2D representation)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to target shape (210, 210)
        
        # Calculate zoom factors for each dimension
        zoom_factors = (target_shape[0] / mel_spec_db.shape[0], 
                       target_shape[1] / mel_spec_db.shape[1])
        
        # Resize the spectrogram
        resized_spec = zoom(mel_spec_db, zoom_factors, order=1)
        
        # Ensure exact target shape
        if resized_spec.shape != target_shape:
            # Pad or crop to exact dimensions
            padded_spec = np.zeros(target_shape)
            min_rows = min(resized_spec.shape[0], target_shape[0])
            min_cols = min(resized_spec.shape[1], target_shape[1])
            padded_spec[:min_rows, :min_cols] = resized_spec[:min_rows, :min_cols]
            resized_spec = padded_spec
        
        print(f"Spectrogram shape: {resized_spec.shape}")
        
        # Normalize the spectrogram to [0, 1] range
        spec_normalized = (resized_spec - resized_spec.min()) / (resized_spec.max() - resized_spec.min() + 1e-8)
        
        # Add channel dimension and batch dimension: (1, 210, 210, 1)
        spec_final = spec_normalized.reshape(1, target_shape[0], target_shape[1], 1)
        
        print(f"Final features shape: {spec_final.shape}")
        return spec_final
        
    except Exception as e:
        print("Failed to process file:", e)
        return None
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"Cleaning up temporary file: {temp_file_path}")
            os.unlink(temp_file_path)

def extract_mfcc_from_mp3(file_path, max_len=130, n_mfcc=20):
    """
    Legacy MFCC extraction function - kept for compatibility
    """
    temp_file_path = None
    try:
        print(f"Processing file (MFCC): {file_path}")
        
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

# === Predict genre ===
def predict_genre_from_mp3(file_path):
    # Check if model is loaded
    if model is None:
        print("Model not loaded - returning demo genre prediction")
        # Return a demo genre index for testing purposes
        import random
        return random.randint(0, 9)  # Random genre for demo (0-9 for 10 genres)
    
    # Try the new spectrogram feature extraction method first
    try:
        features = extract_features_from_mp3(file_path)
        if features is None:
            return "Error extracting features"

        print(f"Features shape: {features.shape}")
        
        # The features are already in the correct format (1, 210, 210, 1)
        # No scaler needed for normalized spectrograms
        
        # Predict
        predictions = model.predict(features)
        predicted_index = np.argmax(predictions)
        
        print(f"Model predictions: {predictions}")
        print(f"Predicted index: {predicted_index}")
        
        # Use encoder if available, otherwise return the index
        if encoder is not None:
            predicted_genre = encoder.inverse_transform([predicted_index])[0]
            return predicted_genre
        else:
            return predicted_index
            
    except Exception as e:
        print(f"Error with new feature extraction: {e}")
        print("Falling back to MFCC extraction...")
        
        # Fallback to MFCC extraction
        try:
            mfcc = extract_mfcc_from_mp3(file_path)
            if mfcc is None:
                return "Error extracting MFCC features"

            # For MFCC, we need to flatten the features to match scaler
            mfcc_flattened = mfcc.flatten().reshape(1, -1)
            print(f"MFCC flattened shape: {mfcc_flattened.shape}")
            
            # If the scaler still expects different dimensions, we might need to pad or truncate
            if mfcc_flattened.shape[1] != 44100:
                if mfcc_flattened.shape[1] < 44100:
                    # Pad with zeros
                    padding = 44100 - mfcc_flattened.shape[1]
                    mfcc_flattened = np.pad(mfcc_flattened, ((0, 0), (0, padding)), mode='constant')
                else:
                    # Truncate
                    mfcc_flattened = mfcc_flattened[:, :44100]
            
            # Normalize
            mfcc_scaled = scaler.transform(mfcc_flattened)

            # Predict
            predictions = model.predict(mfcc_scaled)
            predicted_index = np.argmax(predictions)
            
            if encoder is not None:
                predicted_genre = encoder.inverse_transform([predicted_index])[0]
                return predicted_genre
            else:
                return predicted_index
                
        except Exception as fallback_error:
            print(f"Fallback MFCC extraction also failed: {fallback_error}")
            return f"Error extracting features: {fallback_error}"

def genre_index_to_label(index):
    genre_labels = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock"
    }
    return genre_labels.get(index, "Unknown")

# file_path = r"assets/sampletracks/sample.mp3"
# genre = predict_genre_from_mp3(file_path)
# print("Predicted genre:", genre)
# genre = genre_index_to_label(genre)