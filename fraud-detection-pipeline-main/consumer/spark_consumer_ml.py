import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from pyspark.ml import PipelineModel

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
SPARK_MASTER            = os.getenv("SPARK_MASTER", "local[*]")
OUTPUT_PATH             = os.getenv("OUTPUT_PATH", "/data/output/fraud_flagged")
CHECKPOINT_PATH         = os.getenv("CHECKPOINT_PATH", "/data/checkpoints/fraud_consumer")
PYTHONPATH              = os.getenv("PYTHONPATH", "/app")
# The model is copied into the consumer directory and built into the image at /app/fraud_rf_model
MODEL_PATH              = os.getenv("MODEL_PATH", "/app/fraud_rf_model")

LOCATION_SCHEMA = StructType([
    StructField("city",    StringType()),
    StructField("country", StringType()),
    StructField("lat",     DoubleType()),
    StructField("lon",     DoubleType()),
])

TRANSACTION_SCHEMA = StructType([
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
])

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FraudDetectionConsumerML")
        .config("spark.executorEnv.PYTHONPATH", PYTHONPATH)
        .config("spark.yarn.appMasterEnv.PYTHONPATH", PYTHONPATH)
        .getOrCreate()
    )

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[Consumer] Loading ML model from {MODEL_PATH} ...")
    try:
        model = PipelineModel.load(MODEL_PATH)
        print("[Consumer] ML model loaded successfully.")
    except Exception as e:
        print(f"[Consumer] Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = raw_stream.select(
        from_json(col("value").cast("string"), TRANSACTION_SCHEMA).alias("data")
    ).select("data.*")

    from features.feature_engineering import enrich_transaction

    enriched = enrich_transaction(parsed)

    # Convert structs/booleans to formats the ML Pipeline requires
    ml_ready = enriched.withColumn("is_night_int", col("is_night_transaction").cast("int")) \
                       .withColumn("lat", col("location.lat")) \
                       .withColumn("lon", col("location.lon"))

    # Apply the Random Forest ML Pipeline
    scored = model.transform(ml_ready)

    # Filter flagged frauds (prediction == 1.0) and rename to flagged_fraud
    flagged = scored.withColumn("flagged_fraud", col("prediction") == 1.0) \
                    .filter(col("flagged_fraud") == True)
                    
    # Only keep the final columns per SCHEMA.md
    final_cols = ["transaction_id", "timestamp", "user_id", "card_number", 
                  "merchant", "merchant_category", "amount", "currency", 
                  "location", "is_fraud", "event_time", "hour_of_day", 
                  "is_night_transaction", "amount_bucket", "geo_risk_score", 
                  "flagged_fraud"]
                  
    final_df = flagged.select(*final_cols)

    query = (
        final_df.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", OUTPUT_PATH)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="3 seconds")
        .start()
    )

    print(f"[Consumer] Streaming ML query started (id={query.id}).")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
    except Exception as exc:
        print(f"[Consumer] Streaming query failed: {exc}", file=sys.stderr)
        query.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
