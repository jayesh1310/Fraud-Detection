"""
spark_consumer.py
-----------------
Reads transaction events from Kafka using Spark Structured Streaming,
applies feature engineering, runs a simple rule-based fraud scorer,
and writes flagged transactions to the storage layer (Parquet).
"""

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
SPARK_MASTER            = os.getenv("SPARK_MASTER", "local[*]")
OUTPUT_PATH             = os.getenv("OUTPUT_PATH", "/data/output/fraud_flagged")
CHECKPOINT_PATH         = os.getenv("CHECKPOINT_PATH", "/data/checkpoints/fraud_consumer")
# /app is set via ENV PYTHONPATH in the Dockerfile; propagate it to Spark
# executors so the `features` package is importable on every worker process.
PYTHONPATH              = os.getenv("PYTHONPATH", "/app")

# ---------------------------------------------------------------------------
# Transaction schema (mirrors transaction_producer.py)
# ---------------------------------------------------------------------------
LOCATION_SCHEMA = StructType(
    [
        StructField("city",    StringType()),
        StructField("country", StringType()),
        StructField("lat",     DoubleType()),
        StructField("lon",     DoubleType()),
    ]
)

TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id",    StringType()),
        StructField("timestamp",         StringType()),
        StructField("user_id",           StringType()),
        StructField("card_number",       StringType()),
        StructField("merchant",          StringType()),
        StructField("merchant_category", StringType()),
        StructField("amount",            DoubleType()),
        StructField("currency",          StringType()),
        StructField("location",          LOCATION_SCHEMA),
        StructField("is_fraud",          BooleanType()),
    ]
)

# ---------------------------------------------------------------------------
# Simple rule-based fraud scorer (replace with ML model in production)
# ---------------------------------------------------------------------------
@udf(returnType=BooleanType())
def fraud_score_udf(amount: float, merchant_category: str) -> bool:
    """Flag a transaction as suspicious based on simple heuristics."""
    HIGH_RISK_CATEGORIES = {"electronics", "travel"}
    return amount > 3_000 or merchant_category in HIGH_RISK_CATEGORIES


# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FraudDetectionConsumer")
        # Fix Cause 3: propagate PYTHONPATH to every executor Python worker
        # so `from features.feature_engineering import ...` succeeds on all nodes.
        .config("spark.executorEnv.PYTHONPATH", PYTHONPATH)
        .config("spark.yarn.appMasterEnv.PYTHONPATH", PYTHONPATH)   # YARN compat
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # Ensure output and checkpoint directories exist on the mounted volume
    # before Spark attempts to write to them. exist_ok=True makes this
    # idempotent across container restarts.
    os.makedirs(OUTPUT_PATH,    exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # -----------------------------------------------------------------------
    # 1. Read from Kafka
    # -----------------------------------------------------------------------
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        # Fix Cause 2: use "earliest" so the stream has data immediately on
        # first boot and doesn't exit when the topic tail has no new messages.
        # Switch to "latest" in production once the pipeline is warm.
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")   # tolerate Kafka log compaction
        .load()
    )

    # -----------------------------------------------------------------------
    # 2. Parse JSON payload
    # -----------------------------------------------------------------------
    parsed = raw_stream.select(
        from_json(col("value").cast("string"), TRANSACTION_SCHEMA).alias("data")
    ).select("data.*")

    # -----------------------------------------------------------------------
    # 3. Feature engineering (imported from features/)
    # -----------------------------------------------------------------------
    from features.feature_engineering import enrich_transaction  # noqa: E402

    enriched = enrich_transaction(parsed)

    # -----------------------------------------------------------------------
    # 4. Apply fraud scorer
    # -----------------------------------------------------------------------
    scored = enriched.withColumn(
        "flagged_fraud",
        fraud_score_udf(col("amount"), col("merchant_category")),
    )

    flagged = scored.filter(col("flagged_fraud") == True)  # noqa: E712

    # -----------------------------------------------------------------------
    # 5. Sink to Parquet
    # -----------------------------------------------------------------------
    query = (
        flagged.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", OUTPUT_PATH)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="10 seconds")
        .start()
    )

    print(f"[Consumer] Streaming query started (id={query.id}).")
    print(f"[Consumer] Output  -> {OUTPUT_PATH}")
    print(f"[Consumer] Checkpoint -> {CHECKPOINT_PATH}")
    print("[Consumer] Awaiting termination — press Ctrl+C to stop.")

    # Block the driver process until the query stops or an exception occurs.
    # StreamingQueryException is re-raised here so Docker sees a non-zero
    # exit code and the restart: on-failure policy kicks in correctly.
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("[Consumer] Interrupted — stopping query gracefully.")
        query.stop()
    except Exception as exc:
        print(f"[Consumer] Streaming query failed: {exc}", file=sys.stderr)
        query.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
