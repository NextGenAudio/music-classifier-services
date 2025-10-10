import asyncio
import json
import os
from aiokafka import AIOKafkaConsumer
from aiokafka import AIOKafkaProducer
import logging
from predict_mood_v2 import feature_extractor, predict_mood_from_features, predict_mood_from_mp3

logger = logging.getLogger("my_fastapi_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.propagate = True

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092") # Use service name in Docker
KAFKA_TOPIC_RECIEVED = "audio.uploaded"
KAFKA_TOPIC_PROCESSED = "audio.processed"
GROUP_ID = "audio-service-group"

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
            
            # Use the updated v2 prediction function
            mood = predict_mood_from_mp3(data.get("storageUrl"))
            
            # Alternative: Extract features manually (if needed)
            # feature1 = feature_extractor(data.get("storageUrl")).reshape(1,162) 
            # mood = predict_mood_from_features(feature1)
            
            await send_message(producer=producer, message={"mood": mood,"fileId": data.get("fileId")})
            logger.info(f"Processed message with mood: {mood}")
    finally:
        await consumer.stop()