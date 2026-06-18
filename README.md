# Real-Time Financial Fraud Detection System

This project is an end-to-end Big Data machine learning pipeline for detecting fraudulent financial transactions in real-time. It uses a **custom synthetic data generator** to simulate a live stream of transactions, which are brokered by **Apache Kafka** and processed continuously by an **Apache Spark (PySpark) Structured Streaming** application powered by a **Random Forest Machine Learning model**.

## Architecture Overview

```text
[Synthetic Data Generator] (Faker)
      │ (10 transactions / sec)
      ▼
[Kafka Producer]  ──►  Kafka Topic: transactions
                                │
                                ▼
                      [Spark ML Consumer]
                                │
                     Feature Engineering & 
                 Random Forest Classification
                                │
                                ▼
                     [Parquet Storage Sink]
                        (Fraud Flagged)
```

The system is fully containerized using Docker Compose, orchestrating Zookeeper, Kafka, Spark Master/Worker, the Producer, and the Consumer.

## Project Structure

- `fraud-detection-pipeline-main/`: Contains the core Docker architecture and streaming scripts.
  - `producer/transaction_producer.py`: Generates the live synthetic credit card transactions.
  - `consumer/spark_consumer_ml.py`: The PySpark Structured Streaming consumer that loads the ML model and scores transactions in real-time.
- `train_model_pipeline.py`: The PySpark script used to train the Random Forest ML Pipeline (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, RandomForestClassifier) on a snapshot of the live data.
- `fraud_rf_model/`: The exported PySpark PipelineModel used by the consumer to make real-time predictions.

## Machine Learning Integration

Unlike static batch-processing projects (e.g. using pre-downloaded datasets like PaySim), this project features a model trained directly on a live stream snapshot. This guarantees 100% schema alignment between training and deployment.

The Spark ML Pipeline processes the following features:
- `amount` (StandardScaled)
- `merchant_category` (One-Hot Encoded)
- `amount_bucket` (StringIndexed)
- `hour_of_day` & `is_night_transaction`
- `geo_risk_score` (Derived from Location)

## Usage & Live Demo

Ensure Docker Desktop is running, then follow these steps:

1. **Start the Pipeline**:
   ```bash
   cd fraud-detection-pipeline-main
   docker compose up -d
   ```
2. **Watch the Data Generation**:
   ```bash
   docker logs fraud-producer -f
   ```
3. **Monitor the Spark ML Consumer**:
   ```bash
   docker logs fraud-consumer -f
   ```
4. **View the Spark Dashboard**: Navigate to `http://localhost:4040` in your web browser to watch the streaming micro-batches in real-time.
5. **Analyze the Caught Fraud**: You can copy the Parquet output from the Docker volume to your local machine to view transactions where `flagged_fraud` is True:
   ```bash
   docker cp fraud-consumer:/data/output/fraud_flagged ./temp_flagged
   python -c "import pandas as pd; import glob; files=glob.glob('./temp_flagged/*.parquet'); df=pd.concat([pd.read_parquet(f) for f in files]); print(df[df['flagged_fraud']==True][['transaction_id', 'amount', 'merchant_category', 'is_fraud', 'flagged_fraud']].tail(10))"
   ```
