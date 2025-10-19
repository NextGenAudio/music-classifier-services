import asyncio
import json
from aiokafka import AIOKafkaConsumer
from aiokafka import AIOKafkaProducer
from predict_genre import predict_genre_from_mp3, genre_index_to_label
import logging
import os

logger = logging.getLogger("genre_classifier_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.propagate = True

KAFKA_BOOTSTRAP_SERVERS= os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RECIEVED = "audio.uploaded.genre"
KAFKA_TOPIC_PROCESSED = "audio.processed.genre"
GROUP_ID = "genre-service-group"

async def send_message(producer: AIOKafkaProducer, message: dict):
    """Send a JSON message to Kafka"""
    value_bytes = json.dumps(message).encode("utf-8")
    print(f"Sending message to {KAFKA_TOPIC_PROCESSED}: {value_bytes}")
    await producer.send_and_wait(KAFKA_TOPIC_PROCESSED, value_bytes)

async def consume():
    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC_RECIEVED,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS
    )
    await consumer.start()
    await producer.start()
    try:
        async for msg in consumer:
            logger.info(f"Received message: {msg.value}")
            if isinstance(msg.value, str):
                try:
                    data = json.loads(msg.value)
                except Exception as e:
                    logger.error(f"Failed to decode JSON: {e}")
                    continue
            else:
                data=msg.value
            
            if not isinstance(data,dict):
                logger.error(f"Message value is not a dict: {data}")
                continue
            
            # Get the raw prediction result
            genre_result = predict_genre_from_mp3(data.get("storageUrl"))
            
            # Check if result is already a string (genre label) or int (index)
            if isinstance(genre_result, str):
                # Already a genre label, use directly
                genre = genre_result
            else:
                # It's an index, convert to label
                genre = genre_index_to_label(genre_result)
            
            await send_message(producer=producer, message={"genre": genre,"fileId": data.get("fileId")})
            logger.info(f"Processed message with genre: {genre}")
    finally:
        await consumer.stop()