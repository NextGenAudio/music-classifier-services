from fastapi import FastAPI
import uvicorn
from kafka_consumer import consume
from predict_mood import predict_mood_from_mp3, mood_index_to_label
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.create_task(consume())
    yield  # Application runs here

app = FastAPI(lifespan=lifespan)

@app.get("/getmood")
def read_root(file_path: str = "music_classifier_service/sampletracks/sample.mp3"):
    mood = predict_mood_from_mp3(file_path)
    mood = mood_index_to_label(mood)
    return {"mood": mood}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")