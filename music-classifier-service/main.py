from fastapi import FastAPI
import uvicorn
from predict_mood import predict_mood_from_mp3, mood_index_to_label

app = FastAPI()

@app.get("/getmood")
def read_root(file_path: str):
    mood = predict_mood_from_mp3(file_path)
    mood = mood_index_to_label(mood)
    return {"mood": mood}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")