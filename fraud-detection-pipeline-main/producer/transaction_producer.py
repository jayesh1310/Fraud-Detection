"""
transaction_producer.py
-----------------------
Generates synthetic credit-card transaction events using Faker and
streams them to a Kafka topic in real time.
"""

import json
import os
import random
import time
import uuid
from datetime import datetime

from faker import Faker
from kafka import KafkaProducer

# ---------------------------------------------------------------------------
# Configuration (overridden via environment variables in Docker)
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
TRANSACTIONS_PER_SECOND = int(os.getenv("TRANSACTIONS_PER_SECOND", "10"))
FRAUD_PROBABILITY = float(os.getenv("FRAUD_PROBABILITY", "0.02"))  # 2 % fraud rate

fake = Faker()


def generate_transaction(is_fraud: bool = False) -> dict:
    """Return a single synthetic transaction record."""
    amount = round(random.uniform(5_000, 25_000), 2) if is_fraud else round(random.uniform(1, 2_000), 2)
    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": fake.uuid4(),
        "card_number": fake.credit_card_number(),
        "merchant": fake.company(),
        "merchant_category": random.choice(
            ["retail", "food", "travel", "entertainment", "electronics", "healthcare"]
        ),
        "amount": amount,
        "currency": "USD",
        "location": {
            "city": fake.city(),
            "country": fake.country_code(),
            "lat": float(fake.latitude()),
            "lon": float(fake.longitude()),
        },
        "is_fraud": is_fraud,
    }


def create_producer() -> KafkaProducer:
    """Create and return a configured KafkaProducer."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=5,
    )


def main():
    print(f"[Producer] Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
    producer = create_producer()
    print(f"[Producer] Streaming transactions to topic '{KAFKA_TOPIC}' at {TRANSACTIONS_PER_SECOND} tx/s")

    try:
        while True:
            is_fraud = random.random() < FRAUD_PROBABILITY
            txn = generate_transaction(is_fraud=is_fraud)
            producer.send(KAFKA_TOPIC, value=txn)
            label = "FRAUD" if is_fraud else "legit"
            print(f"[Producer] Sent [{label}] transaction {txn['transaction_id']} — ${txn['amount']}")
            time.sleep(1 / TRANSACTIONS_PER_SECOND)
    except KeyboardInterrupt:
        print("[Producer] Shutting down...")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
