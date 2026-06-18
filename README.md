# Financial Transaction Fraud Detection System

This project is a machine learning pipeline for detecting fraudulent financial transactions using PySpark. It trains a Random Forest Classifier on the PaySim dataset to identify suspicious activities like fraudulent transfers and cash-outs.

## Overview

The system uses PySpark's MLlib to handle large-scale financial data efficiently. The main script trains a model on pre-engineered Parquet files, focusing on class imbalance and evaluating the model's performance via multiple metrics (AUC-ROC, F1 Score, Precision, Recall).

## Files and Structure

- `eda_paysim.ipynb`: Exploratory Data Analysis notebook to analyze the PaySim dataset.
- `feature_engineering.py`: Script to process the raw dataset and extract meaningful features.
- `train_model.py`: PySpark script that trains the Random Forest model and evaluates it.
- `train_model_pipeline.py`: Pipeline for end-to-end model training.
- `fraud_rf_model/`: Directory where the trained PySpark Random Forest model is saved.
- `training_data/`: Contains intermediate data/Parquet files.

## Features Used

The model uses the following engineered features to detect fraud:
- Transaction amount
- Old and new balances of origin and destination accounts
- Balance differences
- Discrepancies (errors) in expected balances

## Usage

1. **Setup**: Ensure you have Java, Spark, and PySpark installed. On Windows, `winutils.exe` is required.
2. **Train Model**: Run the training script:
   ```bash
   python train_model.py
   ```
3. **Deploy**: The model is saved to the `fraud_rf_model` folder and can be loaded for real-time predictions in a streaming pipeline (e.g., `spark_consumer_ml.py`).

## Evaluation

The model provides metrics such as AUC-ROC, Precision, and Recall, focusing specifically on minimizing False Negatives (missed frauds) while maintaining a low false alarm rate.
