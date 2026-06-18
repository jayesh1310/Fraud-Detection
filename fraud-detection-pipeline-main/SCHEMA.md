# Transaction & Pipeline Schema Reference

> **Audience**: ML engineers building an Apache Spark MLlib model on top of the  
> fraud-detection pipeline. This document is the single source of truth for all  
> data shapes — from raw Kafka events through feature enrichment to the final  
> Parquet training dataset.

---

## Table of Contents

1. [Kafka Topic: `transactions` (raw)](#1-kafka-topic-transactions-raw)
2. [Kafka Topic: `enriched_transactions`](#2-kafka-topic-enriched_transactions)
3. [HDFS / Local Parquet Output Structure](#3-hdfs--local-parquet-output-structure)
4. [Feature Column Reference for MLlib](#4-feature-column-reference-for-mllib)
5. [Data Types Quick-Reference](#5-data-types-quick-reference)
6. [Schema Evolution & Versioning Notes](#6-schema-evolution--versioning-notes)

---

## 1. Kafka Topic: `transactions` (raw)

**Topic name**: `transactions`  
**Producer**: `producer/transaction_producer.py`  
**Encoding**: UTF-8 JSON, one record per Kafka message  
**Partition key**: none (round-robin)  
**Throughput**: configurable via `TRANSACTIONS_PER_SECOND` (default 10 msg/s)

### JSON Schema

```json
{
  "transaction_id":    "<uuid-v4 string>",
  "timestamp":         "<ISO-8601 UTC string>  e.g. 2024-06-16T11:42:00.123456",
  "user_id":           "<uuid-v4 string>",
  "card_number":       "<string>  masked 16-digit card number",
  "merchant":          "<string>  company name",
  "merchant_category": "<string>  one of: retail | food | travel | entertainment | electronics | healthcare",
  "amount":            "<float>   USD, 2 decimal places>",
  "currency":          "<string>  always 'USD' in v1>",
  "location": {
    "city":    "<string>",
    "country": "<ISO 3166-1 alpha-2 string>  e.g. 'IN', 'US'",
    "lat":     "<float>  WGS-84 latitude  -90.0 to 90.0",
    "lon":     "<float>  WGS-84 longitude -180.0 to 180.0"
  },
  "is_fraud":          "<boolean>  ground-truth label (from simulator)"
}
```

### Field Details

| Field | Type | Nullable | Notes |
|-------|------|----------|-------|
| `transaction_id` | `string` | No | UUID v4, globally unique |
| `timestamp` | `string` | No | UTC, microsecond precision |
| `user_id` | `string` | No | UUID v4, stable per simulated user |
| `card_number` | `string` | No | Faker-generated, NOT real PAN |
| `merchant` | `string` | No | Free-text company name |
| `merchant_category` | `string` | No | One of 6 fixed categories (see above) |
| `amount` | `float64` | No | Fraud txns: $5,000–$25,000 / Legit: $1–$2,000 |
| `currency` | `string` | No | Always `"USD"` in v1 |
| `location.city` | `string` | No | Faker city name |
| `location.country` | `string` | No | ISO 3166-1 alpha-2 |
| `location.lat` | `float64` | No | WGS-84 |
| `location.lon` | `float64` | No | WGS-84 |
| `is_fraud` | `boolean` | No | **Label column** — 2% fraud rate in simulator |

### Example Message

```json
{
  "transaction_id": "3f7a1c2e-9b4d-4e8f-a1b2-c3d4e5f60001",
  "timestamp": "2024-06-16T05:48:22.341201",
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "card_number": "4532015112830366",
  "merchant": "Peterson & Sons Ltd",
  "merchant_category": "electronics",
  "amount": 8450.75,
  "currency": "USD",
  "location": {
    "city": "Mumbai",
    "country": "IN",
    "lat": 19.0760,
    "lon": 72.8777
  },
  "is_fraud": true
}
```

---

## 2. Kafka Topic: `enriched_transactions`

> **Note**: In the current pipeline, enrichment happens inside the Spark consumer  
> and results are written directly to Parquet — not re-published to a separate  
> Kafka topic. If you add a second Kafka topic for downstream consumers, use the  
> schema below as the contract.

**Topic name**: `enriched_transactions` *(planned)*  
**Producer**: `consumer/spark_consumer.py` (after feature engineering)  
**Encoding**: UTF-8 JSON or Avro (recommended for production)

### JSON Schema (extends `transactions`)

```json
{
  // ── All fields from `transactions` topic (see §1) ──────────────────────────

  "event_time":            "<timestamp>  parsed from `timestamp` string",
  "hour_of_day":           "<integer>    0–23",
  "is_night_transaction":  "<boolean>    true if hour_of_day in [22..23, 0..5]",
  "amount_bucket":         "<string>     Low | Medium | High | Very High",
  "geo_risk_score":        "<float>      0.5 | 1.0 | 2.0",
  "flagged_fraud":         "<boolean>    output of rule-based scorer"
}
```

### Enriched Field Details

| Field | Type | Nullable | Source | Description |
|-------|------|----------|--------|-------------|
| `event_time` | `timestamp` | No | `feature_engineering.py` | Parsed from `timestamp` via `to_timestamp()` |
| `hour_of_day` | `integer` | No | `feature_engineering.py` | Hour extracted from `event_time` (0–23) |
| `is_night_transaction` | `boolean` | No | `feature_engineering.py` | `true` when `hour_of_day >= 22 OR <= 5` |
| `amount_bucket` | `string` | No | `feature_engineering.py` | `< $100` → Low, `$100–$499` → Medium, `$500–$1,999` → High, `>= $2,000` → Very High |
| `geo_risk_score` | `float64` | No | `feature_engineering.py` | `abs(lat) > 60` → 2.0, `> 40` → 1.0, else 0.5 |
| `flagged_fraud` | `boolean` | No | `spark_consumer.py` | Rule: `amount > $3,000 OR category in {electronics, travel}` |

---

## 3. HDFS / Local Parquet Output Structure

**Output base path**: `/data/output/fraud_flagged`  
**Sink mode**: Append (Spark Structured Streaming)  
**Trigger**: every 10 seconds  
**Format**: Apache Parquet (Snappy compressed by default)  
**Partitioning**: none in v1 (add `date` partition for production)

### Directory Layout

```
/data/output/
  fraud_flagged/
    part-00000-<uuid>.snappy.parquet
    part-00001-<uuid>.snappy.parquet
    ...
  checkpoints/
    fraud_consumer/
      commits/
      offsets/
      sources/
```

### Parquet Schema (Spark DDL notation)

```sql
root
 |-- transaction_id:        string    (nullable = false)
 |-- timestamp:             string    (nullable = false)
 |-- user_id:               string    (nullable = false)
 |-- card_number:           string    (nullable = false)
 |-- merchant:              string    (nullable = false)
 |-- merchant_category:     string    (nullable = false)
 |-- amount:                double    (nullable = false)
 |-- currency:              string    (nullable = false)
 |-- location:              struct    (nullable = true)
 |    |-- city:             string    (nullable = true)
 |    |-- country:          string    (nullable = true)
 |    |-- lat:              double    (nullable = true)
 |    |-- lon:              double    (nullable = true)
 |-- is_fraud:              boolean   (nullable = false)   ← LABEL
 |-- event_time:            timestamp (nullable = true)
 |-- hour_of_day:           integer   (nullable = true)
 |-- is_night_transaction:  boolean   (nullable = true)
 |-- amount_bucket:         string    (nullable = true)
 |-- geo_risk_score:        double    (nullable = true)
 |-- flagged_fraud:         boolean   (nullable = false)   ← RULE-BASED PREDICTION
```

### Reading the Parquet Files (PySpark)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FraudMLModel").getOrCreate()

df = spark.read.parquet("/data/output/fraud_flagged")
df.printSchema()
df.show(5)
```

### Reading with Pandas (for local EDA)

```python
import pandas as pd

df = pd.read_parquet("/data/output/fraud_flagged/")
print(df.dtypes)
print(df["is_fraud"].value_counts())
```

---

## 4. Feature Column Reference for MLlib

Use this table to build your `VectorAssembler` feature pipeline.

| Column | Spark Type | Role | MLlib Notes |
|--------|-----------|------|-------------|
| `amount` | `DoubleType` | Numeric feature | Scale with `StandardScaler` |
| `hour_of_day` | `IntegerType` | Numeric feature | Cyclic — consider `sin/cos` encoding |
| `is_night_transaction` | `BooleanType` | Binary feature | Cast to `IntegerType` (0/1) |
| `geo_risk_score` | `DoubleType` | Ordinal feature | Values: 0.5, 1.0, 2.0 |
| `merchant_category` | `StringType` | Categorical feature | Use `StringIndexer` + `OneHotEncoder` |
| `amount_bucket` | `StringType` | Ordinal categorical | Use `StringIndexer` (ordered) |
| `location.lat` | `DoubleType` | Numeric feature | Consider binning by region |
| `location.lon` | `DoubleType` | Numeric feature | Consider binning by region |
| `is_fraud` | `BooleanType` | **Label** | Cast to `IntegerType`; use `0`/`1` |

### Recommended VectorAssembler Setup

```python
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Cast booleans to integers
df = df.withColumn("label",                col("is_fraud").cast("int")) \
       .withColumn("is_night_int",         col("is_night_transaction").cast("int"))

# Flatten struct column
df = df.withColumn("lat", col("location.lat")) \
       .withColumn("lon", col("location.lon"))

# Encode merchant_category
cat_indexer  = StringIndexer(inputCol="merchant_category", outputCol="cat_idx")
cat_encoder  = OneHotEncoder(inputCol="cat_idx", outputCol="cat_vec")

# Encode amount_bucket (keep ordering: Low=0, Medium=1, High=2, Very High=3)
bucket_order   = ["Low", "Medium", "High", "Very High"]
bucket_indexer = StringIndexer(
    inputCol="amount_bucket", outputCol="bucket_idx",
    stringOrderType="alphabetAsc"   # override manually if needed
)

assembler = VectorAssembler(
    inputCols=[
        "amount", "hour_of_day", "is_night_int",
        "geo_risk_score", "lat", "lon",
        "cat_vec", "bucket_idx"
    ],
    outputCol="raw_features"
)

scaler = StandardScaler(inputCol="raw_features", outputCol="features")

pipeline = Pipeline(stages=[cat_indexer, cat_encoder, bucket_indexer, assembler, scaler])
model    = pipeline.fit(df)
train_df = model.transform(df)
```

---

## 5. Data Types Quick-Reference

| JSON Type | Spark SQL Type | Pandas dtype | Notes |
|-----------|---------------|--------------|-------|
| `string` | `StringType` | `object` | |
| `float` | `DoubleType` | `float64` | All monetary amounts |
| `boolean` | `BooleanType` | `bool` | `is_fraud`, `flagged_fraud`, `is_night_transaction` |
| `integer` | `IntegerType` | `int32` | `hour_of_day` |
| `timestamp` | `TimestampType` | `datetime64[ns]` | `event_time` |
| `struct` | `StructType` | `dict` / flattened cols | `location` |

---

## 6. Schema Evolution & Versioning Notes

> [!IMPORTANT]
> Before making **any breaking change** to a Kafka topic schema, coordinate with
> all downstream consumers (Spark streaming jobs, ML training pipelines, dashboards).

| Change Type | Impact | Recommended Approach |
|-------------|--------|----------------------|
| Add new nullable field | Low | Safe — consumers ignore unknown fields |
| Rename existing field | **Breaking** | Version the topic (`transactions_v2`) or use Avro schema registry |
| Change field type | **Breaking** | Same as rename — version the topic |
| Remove field | **Breaking** | Deprecate first, remove after all consumers updated |
| Add new `merchant_category` value | Medium | Retrain `StringIndexer` models; update `OneHotEncoder` output size |

### Recommended Production Upgrade: Avro + Schema Registry

```
producer/ → Confluent Schema Registry → Kafka → consumer/
```

- Register schemas at `http://schema-registry:8081`
- Use `confluent-kafka-python` with `AvroSerializer` / `AvroDeserializer`
- Enforces backward/forward compatibility automatically

---

*Last updated: 2026-06-16 | Pipeline version: v1.0 | Maintainer: fraud-detection-team*
