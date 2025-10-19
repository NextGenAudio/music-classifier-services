from fastapi import FastAPI
import uvicorn
from kafka_consumer import consume
from predict_genre import predict_genre_from_mp3, genre_index_to_label
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.create_task(consume())
    yield  # Application runs here

app = FastAPI(lifespan=lifespan)

@app.get("/getgenre")
def read_root(file_path: str = r"/app/sampletracks/sample1.wav"):
    genre = predict_genre_from_mp3(file_path)
    genre = genre_index_to_label(genre)
    return {"genre": genre}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, log_level="info")