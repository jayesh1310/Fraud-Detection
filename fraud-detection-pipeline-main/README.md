# Fraud Detection Data Pipeline

A real-time Big Data fraud detection pipeline built with **Apache Kafka**, **Apache Spark Structured Streaming**, **PySpark**, and **Faker** — fully orchestrated with **Docker Compose**.

---

## Architecture

```
[Faker Generator]
      │
      ▼
[Kafka Producer]  ──►  Kafka Topic: transactions
                                │
                                ▼
                      [Spark Consumer]
                         │
                    Feature Engineering
                    (features/)
                         │
                    Rule-Based Scorer
                         │
                         ▼
                  [Parquet Sink]
                  (storage/output/)
```

---

## Project Structure

```
fraud-detection-pipeline/
├── docker-compose.yml          # Orchestrates all services
├── requirements.txt            # Root-level dependencies
│
├── producer/
│   ├── __init__.py
│   ├── transaction_producer.py # Kafka producer (Faker-generated events)
│   ├── Dockerfile
│   └── requirements.txt
│
├── consumer/
│   ├── __init__.py
│   ├── spark_consumer.py       # Spark Structured Streaming consumer
│   ├── Dockerfile
│   └── requirements.txt
│
├── features/
│   ├── __init__.py
│   └── feature_engineering.py  # PySpark feature transforms
│
└── storage/
    ├── __init__.py
    └── parquet_sink.py         # Batch & streaming Parquet sinks
```

---

## Services (docker-compose.yml)

| Service        | Image                           | Port(s)        | Purpose                          |
|----------------|---------------------------------|----------------|----------------------------------|
| `zookeeper`    | confluentinc/cp-zookeeper:7.5.0 | 2181           | Kafka coordination               |
| `kafka`        | confluentinc/cp-kafka:7.5.0     | 9092           | Message broker                   |
| `spark`        | bitnami/spark:3.5               | 8080, 7077     | Spark master                     |
| `spark-worker` | bitnami/spark:3.5               | —              | Spark worker (2 cores, 2 GB RAM) |
| `producer`     | ./producer                      | —              | Transaction event generator      |
| `consumer`     | ./consumer                      | —              | Streaming fraud detector         |

---

## Quick Start

### Prerequisites
- Docker Desktop (with Docker Compose v2)
- Python 3.11+ (for local development)

### Run the full pipeline

```bash
# 1. Start all services
docker compose up --build

# 2. View producer logs
docker logs -f fraud-producer

# 3. View consumer / Spark logs
docker logs -f fraud-consumer

# 4. Spark UI
open http://localhost:8080
```

### Local development (without Docker)

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start Kafka locally (requires Kafka installed)
# Then run:
python producer/transaction_producer.py
```

---

## Environment Variables

| Variable                  | Default          | Description                       |
|---------------------------|------------------|-----------------------------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address              |
| `KAFKA_TOPIC`             | `transactions`   | Kafka topic name                  |
| `TRANSACTIONS_PER_SECOND` | `10`             | Producer throughput               |
| `FRAUD_PROBABILITY`       | `0.02`           | Probability of generating a fraud event |
| `SPARK_MASTER`            | `local[*]`       | Spark master URL                  |
| `OUTPUT_PATH`             | `/data/output/fraud_flagged` | Parquet output path  |

---

## Feature Engineering (`features/feature_engineering.py`)

| Feature               | Description                                      |
|-----------------------|--------------------------------------------------|
| `event_time`          | Parsed timestamp (TimestampType)                 |
| `hour_of_day`         | Hour extracted from event_time (0–23)            |
| `is_night_transaction`| True if transaction occurred between 22:00–05:00 |
| `amount_bucket`       | Low / Medium / High / Very High                  |
| `geo_risk_score`      | 0.5 / 1.0 / 2.0 based on latitude               |

---

## Extending the Pipeline

- **Add an ML model**: Replace `fraud_score_udf` in `consumer/spark_consumer.py` with a PySpark MLlib model or ONNX inference.
- **Add more features**: Drop new transform functions into `features/feature_engineering.py` and chain them in `enrich_transaction()`.
- **Change the sink**: Use `storage/parquet_sink.py` helpers or swap Parquet for Delta Lake / Apache Iceberg.
- **Scale**: Increase `SPARK_WORKER_CORES` / `SPARK_WORKER_MEMORY` in `docker-compose.yml`.
