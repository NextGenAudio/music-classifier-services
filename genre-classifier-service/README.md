# Genre Classifier Service

A microservice for classifying music genres using machine learning. This service integrates with Kafka for real-time audio processing and provides REST API endpoints for genre prediction.

## Features

- Real-time genre classification from audio files
- Kafka integration for event-driven processing
- Support for various audio formats (MP3, WAV, etc.)
- S3 and HTTPS URL support for remote files
- RESTful API for direct genre prediction
- Docker containerization support

## Genre Categories

The service can classify music into the following genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## API Endpoints

### GET /getgenre
Predict the genre of an audio file.

**Parameters:**
- `file_path` (optional): Path to the audio file. Defaults to sample track.

**Response:**
```json
{
  "genre": "rock"
}
```

## Kafka Integration

The service listens to the `audio.uploaded` topic and publishes results to the `audio.processed` topic.

**Input Message Format:**
```json
{
  "fileId": "12345",
  "storageUrl": "https://s3.amazonaws.com/bucket/audio.mp3"
}
```

**Output Message Format:**
```json
{
  "genre": "rock",
  "fileId": "12345"
}
```

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Kafka:
```bash
docker-compose up -d
```

3. Run the service:
```bash
python main.py
```

The service will be available at `http://localhost:8002`

## Model Files

Place your trained genre classification model files in the `assets/` directory:
- `genre_model.h5` - Trained Keras model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder for genre classes

## Development

The service uses:
- FastAPI for the web framework
- Kafka for message streaming
- TensorFlow/Keras for ML inference
- Librosa for audio feature extraction
- Boto3 for S3 integration