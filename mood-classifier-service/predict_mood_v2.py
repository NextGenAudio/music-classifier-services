import numpy as np
import librosa  
from tensorflow.keras.models import load_model
import joblib

model=load_model(r"D:\Projects\Sonex\music-classifier-service\music-classifier-service\assets\mood_model_v2.h5") 
encoder = joblib.load(r"D:\Projects\Sonex\music-classifier-service\music-classifier-service\assets\label_encoder_v2.pkl")

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

def predict_decode(feature):
    predict = np.argmax(model.predict(feature))
    predict_encode = np.array([])
    for i in range(5):
        if i == predict:
            predict_encode = np.append(predict_encode,[1])
        else:
            predict_encode = np.append(predict_encode,[0])
    predict_encode = predict_encode.reshape(1,-1)
    predict_decode = encoder.inverse_transform(predict_encode)
    return predict_decode[0][0]

feature1 = feature_extractor(r"D:\Projects\Sonex\music-classifier-service\music-classifier-service\sampletracks\sample.mp3").reshape(1,162)  
print("MyGO!!!!! Hekitenbansou:")
print("classification:",predict_decode(feature1))
print("-------------------------------------")
