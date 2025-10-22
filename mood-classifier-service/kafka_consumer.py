import asyncio
import json
import os
import socket
from aiokafka import AIOKafkaConsumer
from aiokafka import AIOKafkaProducer
import logging
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from predict_mood_v2 import feature_extractor, predict_mood_from_features, predict_mood_from_mp3

logger = logging.getLogger("my_fastapi_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.propagate = True

class MSKTokenProvider():
    def token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token('us-east-1')
        return token

KAFKA_BOOTSTRAP_SERVERS = "boot-ctavlxrz.c2.kafka-serverless.us-east-1.amazonaws.com:9098"
KAFKA_TOPIC_RECIEVED = "audio.uploaded.mood"
KAFKA_TOPIC_PROCESSED = "audio.processed.mood"
GROUP_ID = "mood-service-group"

# Kafka security configuration
KAFKA_SECURITY_CONFIG = {
    'security_protocol': 'SASL_SSL',
    'sasl_mechanism': 'OAUTHBEARER',
    'sasl_oauth_token_provider': MSKTokenProvider(),
    'client_id': socket.gethostname()
}

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
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        **KAFKA_SECURITY_CONFIG
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        **KAFKA_SECURITY_CONFIG
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