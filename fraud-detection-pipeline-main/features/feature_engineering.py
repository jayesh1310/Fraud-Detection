"""
feature_engineering.py
-----------------------
PySpark feature-engineering transformations applied to the transaction stream
before fraud scoring.  All functions accept and return a Spark DataFrame so
they compose cleanly inside a Structured Streaming pipeline.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    abs as spark_abs,
    col,
    hour,
    to_timestamp,
    when,
)


def parse_timestamps(df: DataFrame) -> DataFrame:
    """Cast the ISO-8601 timestamp string to a proper TimestampType."""
    return df.withColumn("event_time", to_timestamp(col("timestamp")))


def add_hour_of_day(df: DataFrame) -> DataFrame:
    """Extract the hour of day from event_time (0-23)."""
    return df.withColumn("hour_of_day", hour(col("event_time")))


def add_is_night_transaction(df: DataFrame) -> DataFrame:
    """Flag transactions that occur between 22:00 and 05:00 as night transactions."""
    return df.withColumn(
        "is_night_transaction",
        when((col("hour_of_day") >= 22) | (col("hour_of_day") <= 5), True).otherwise(False),
    )


def add_amount_bucket(df: DataFrame) -> DataFrame:
    """Bucket transaction amounts into Low / Medium / High / Very High tiers."""
    return df.withColumn(
        "amount_bucket",
        when(col("amount") < 100, "Low")
        .when(col("amount") < 500, "Medium")
        .when(col("amount") < 2000, "High")
        .otherwise("Very High"),
    )


def add_geo_risk(df: DataFrame) -> DataFrame:
    """
    Assign a simple geo-risk score based on absolute latitude.
    Higher latitudes (> 60°) are treated as higher risk for this demo.
    """
    return df.withColumn(
        "geo_risk_score",
        when(spark_abs(col("location.lat")) > 60, 2.0)
        .when(spark_abs(col("location.lat")) > 40, 1.0)
        .otherwise(0.5),
    )


def enrich_transaction(df: DataFrame) -> DataFrame:
    """
    Apply the full feature-engineering pipeline to a transaction DataFrame.

    Steps:
      1. Parse timestamps
      2. Add hour-of-day
      3. Flag night transactions
      4. Bucket transaction amounts
      5. Assign geo-risk score

    Returns:
        Enriched DataFrame with additional feature columns.
    """
    df = parse_timestamps(df)
    df = add_hour_of_day(df)
    df = add_is_night_transaction(df)
    df = add_amount_bucket(df)
    df = add_geo_risk(df)
    return df
