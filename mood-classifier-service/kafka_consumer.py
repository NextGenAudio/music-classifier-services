import asyncio
import json
from aiokafka import AIOKafkaConsumer
from aiokafka import AIOKafkaProducer
from predict_mood import predict_mood_from_mp3, mood_index_to_label
import logging

logger = logging.getLogger("my_fastapi_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.propagate = True

KAFKA_BOOTSTRAP_SERVERS= "localhost:9092"
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
            mood = predict_mood_from_mp3(data.get("storageUrl"))   # call your function
            mood = mood_index_to_label(mood)
            await send_message(producer=producer, message={"mood": mood,"fileId": data.get("fileId")})
            logger.info(f"Processed message with mood: {mood}")
    finally:
        await consumer.stop()