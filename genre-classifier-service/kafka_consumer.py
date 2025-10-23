import asyncio
import json
import socket
import ssl
from aiokafka import AIOKafkaConsumer
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError
from predict_genre import predict_genre_from_mp3, genre_index_to_label
import logging
import os
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

logger = logging.getLogger("genre_classifier_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.propagate = True

from aiokafka.abc import AbstractTokenProvider

class MSKTokenProvider(AbstractTokenProvider):
    async def token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token('us-east-1')
        return token

KAFKA_BOOTSTRAP_SERVERS = "boot-ctavlxrz.c2.kafka-serverless.us-east-1.amazonaws.com:9098"
KAFKA_TOPIC_RECIEVED = "audio.uploaded.genre"
KAFKA_TOPIC_PROCESSED = "audio.processed.genre"
GROUP_ID = "genre-service-group"

# Create SSL context for Kafka
ssl_context = ssl.create_default_context()

# Kafka security configuration
KAFKA_SECURITY_CONFIG = {
    'security_protocol': 'SASL_SSL',
    'sasl_mechanism': 'OAUTHBEARER',
    'sasl_oauth_token_provider': MSKTokenProvider(),
    'client_id': socket.gethostname(),
    'ssl_context': ssl_context
}

async def create_kafka_topics():
    """Create Kafka topics if they don't exist"""
    admin_client = AIOKafkaAdminClient(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        **KAFKA_SECURITY_CONFIG
    )
    
    try:
        await admin_client.start()
        logger.info("Admin client connected to Kafka")
        
        # Get existing topics
        metadata = await admin_client.list_topics()
        existing_topics = metadata
        
        topics_to_create = []
        
        # Check and add KAFKA_TOPIC_RECIEVED if it doesn't exist
        if KAFKA_TOPIC_RECIEVED not in existing_topics:
            topics_to_create.append(
                NewTopic(
                    name=KAFKA_TOPIC_RECIEVED,
                    num_partitions=3,
                    replication_factor=2  # MSK Serverless will manage this automatically
                )
            )
            logger.info(f"Will create topic: {KAFKA_TOPIC_RECIEVED}")
        else:
            logger.info(f"✓ Topic already exists: {KAFKA_TOPIC_RECIEVED}")
        
        # Check and add KAFKA_TOPIC_PROCESSED if it doesn't exist
        if KAFKA_TOPIC_PROCESSED not in existing_topics:
            topics_to_create.append(
                NewTopic(
                    name=KAFKA_TOPIC_PROCESSED,
                    num_partitions=3,
                    replication_factor=2  # MSK Serverless will manage this automatically
                )
            )
            logger.info(f"Will create topic: {KAFKA_TOPIC_PROCESSED}")
        else:
            logger.info(f"✓ Topic already exists: {KAFKA_TOPIC_PROCESSED}")
        
        # Create topics only if there are any to create
        if topics_to_create:
            await admin_client.create_topics(topics_to_create)
            logger.info(f"✓ Successfully created {len(topics_to_create)} topic(s)")
            
            # Wait for topics to be available in cluster metadata
            logger.info("Waiting for topics to be available in cluster...")
            await asyncio.sleep(5)
        else:
            logger.info("✓ All topics already exist, nothing to create")
        
        return True
        
    except Exception as e:
        logger.error(f"Topic creation failed: {e}")
        return False
    finally:
        await admin_client.close()

async def wait_for_topic(topic_name: str, max_retries: int = 10, delay: int = 3):
    """Wait for a topic to be available in cluster metadata"""
    for attempt in range(max_retries):
        admin_client = AIOKafkaAdminClient(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            **KAFKA_SECURITY_CONFIG
        )
        
        try:
            await admin_client.start()
            metadata = await admin_client.list_topics()
            
            if topic_name in metadata:
                logger.info(f"✓ Topic {topic_name} is now available in cluster")
                await admin_client.close()
                return True
            
            logger.info(f"Waiting for topic {topic_name}... (attempt {attempt + 1}/{max_retries})")
            await admin_client.close()
            await asyncio.sleep(delay)
            
        except Exception as e:
            logger.warning(f"Error checking topic: {e}")
            if admin_client:
                await admin_client.close()
            await asyncio.sleep(delay)
    
    logger.error(f"Topic {topic_name} not available after {max_retries} attempts")
    return False

async def send_message(producer: AIOKafkaProducer, message: dict):
    """Send a JSON message to Kafka"""
    value_bytes = json.dumps(message).encode("utf-8")
    logger.info(f"Sending message to {KAFKA_TOPIC_PROCESSED}: {message}")
    await producer.send_and_wait(KAFKA_TOPIC_PROCESSED, value_bytes)

async def consume():
    # Create topics before starting consumer
    logger.info("Creating Kafka topics if needed...")
    await create_kafka_topics()
    
     # Wait for the receive topic to be available
    logger.info(f"Verifying topic {KAFKA_TOPIC_RECIEVED} is available...")
    topic_ready = await wait_for_topic(KAFKA_TOPIC_RECIEVED)
    
    if not topic_ready:
        logger.error(f"Topic {KAFKA_TOPIC_RECIEVED} not available. Exiting.")
        return
    
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
        await producer.stop()
        logger.info("Consumer and Producer stopped")