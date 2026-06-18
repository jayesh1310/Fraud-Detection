# -*- coding: utf-8 -*-
# ============================================================
# train_model.py
# Task 3 - MLlib Model Training (RandomForest)

# ---------------------------------------------------------
# WINDOWS FIX: Set HADOOP_HOME before importing PySpark.
# PySpark on Windows needs winutils.exe to write files.
# winutils.exe must be at C:\hadoop\bin\winutils.exe
# (already downloaded by the setup step)
# ---------------------------------------------------------
import os
os.environ["HADOOP_HOME"]       = r"C:\hadoop"
os.environ["hadoop.home.dir"]   = r"C:\hadoop"
# Also add hadoop bin to PATH so hadoop.dll is found by the JVM
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
# ---------------------------------------------------------
#
# Project : Financial Transaction Fraud Detection System
# Author  : Member 2
# Dataset : PaySim - pre-engineered Parquet files
#
# What this script does:
#   1. Starts a local SparkSession
#   2. Loads train_features.parquet & test_features.parquet
#   3. Assembles features with VectorAssembler
#   4. Trains RandomForestClassifier (100 trees, class weights)
#   5. Evaluates: AUC-ROC, F1, Precision, Recall, confusion matrix
#   6. Prints ranked feature importances
#   7. Saves the model to ./fraud_rf_model/
#
# Usage:
#   python train_model.py
#
# Tested with: PySpark 4.1.2, Python 3.12, Java 21
# ============================================================

import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col as fcol
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
TRAIN_PATH = "train_features.parquet"
TEST_PATH  = "test_features.parquet"
MODEL_PATH = "fraud_rf_model"

# -------------------------------------------------------------
# Feature columns (from feature_engineering.py output)
# -------------------------------------------------------------
FEATURE_COLS = [
    "step",               # hour of simulation (time feature)
    "type",               # encoded: 1=TRANSFER, 2=CASH_OUT
    "amount",             # scaled transaction amount
    "oldbalanceOrg",      # scaled sender balance before
    "newbalanceOrig",     # scaled sender balance after
    "oldbalanceDest",     # scaled receiver balance before
    "newbalanceDest",     # scaled receiver balance after
    "balance_diff_orig",  # scaled: sender balance change
    "balance_diff_dest",  # scaled: receiver balance change
    "error_orig",         # sender discrepancy (amount vs balance change)
    "error_dest",         # receiver discrepancy
]

LABEL_COL  = "isFraud"
WEIGHT_COL = "classWeight"
VECTOR_COL = "features"

# -------------------------------------------------------------
# STEP 1 - Start SparkSession
# -------------------------------------------------------------
print("=" * 60)
print("  TASK 3 - PySpark MLlib Model Training")
print("=" * 60)

print("\n[Step 1] Starting SparkSession ...")
spark = (
    SparkSession.builder
    .appName("FraudDetection_RandomForest")
    .master("local[*]")                              # use all CPU cores on your machine
    .config("spark.driver.memory", "4g")             # 4 GB RAM for driver
    .config("spark.sql.shuffle.partitions", "8")     # fewer partitions for local mode
    .config("spark.ui.showConsoleProgress", "false") # cleaner console output
    # --- Windows fixes ---
    # Committer v2 avoids getAllCommittedTaskPaths() which triggers
    # NativeIO$Windows.access0 (the DLL JNI mismatch crash on Windows)
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    # Disable Hadoop native IO entirely - use pure Java fallback
    .config("spark.hadoop.io.native.lib.available", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"  [OK] Spark version : {spark.version}")
print(f"  [OK] Master        : {spark.sparkContext.master}")

# -------------------------------------------------------------
# STEP 2 - Load Train Parquet
# -------------------------------------------------------------
print(f"\n[Step 2] Loading {TRAIN_PATH} ...")
t0 = time.time()
train_df = spark.read.parquet(TRAIN_PATH)
train_df.cache()
train_count = train_df.count()
print(f"  [OK] Train rows : {train_count:,}  (took {time.time()-t0:.1f}s)")

# -------------------------------------------------------------
# STEP 3 - Load Test Parquet
# -------------------------------------------------------------
print(f"\n[Step 3] Loading {TEST_PATH} ...")
t0 = time.time()
test_df = spark.read.parquet(TEST_PATH)
test_df.cache()
test_count = test_df.count()
print(f"  [OK] Test rows  : {test_count:,}  (took {time.time()-t0:.1f}s)")

# Print class balance in train set
fraud_count = train_df.filter(fcol(LABEL_COL) == 1).count()
legit_count  = train_count - fraud_count
print(f"\n  Train set class balance:")
print(f"    Legit (0) : {legit_count:,}  ({legit_count/train_count*100:.2f}%)")
print(f"    Fraud (1) : {fraud_count:,}    ({fraud_count/train_count*100:.4f}%)")

print("\n  Train schema:")
train_df.printSchema()

# -------------------------------------------------------------
# STEP 4 - Assemble Feature Vector
# -------------------------------------------------------------
print(f"\n[Step 4] Assembling features with VectorAssembler ...")
assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol=VECTOR_COL,
    handleInvalid="skip"
)
train_vec = assembler.transform(train_df)
test_vec  = assembler.transform(test_df)

print(f"  [OK] Feature vector column: '{VECTOR_COL}'")
print(f"  [OK] Feature count        : {len(FEATURE_COLS)}")
print(f"  [OK] Features used        : {FEATURE_COLS}")

# -------------------------------------------------------------
# STEP 5 - Train RandomForestClassifier
# -------------------------------------------------------------
print(f"\n[Step 5] Training RandomForestClassifier ...")
print(f"  numTrees  = 100")
print(f"  maxDepth  = 10")
print(f"  weightCol = '{WEIGHT_COL}'  (handles class imbalance - pre-computed in Task 2)")
print(f"  labelCol  = '{LABEL_COL}'")
print(f"\n  Training ... (this takes 2-10 minutes depending on your machine) ...")

t0 = time.time()
rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=10,
    featuresCol=VECTOR_COL,
    labelCol=LABEL_COL,
    weightCol=WEIGHT_COL,
    seed=42,
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
)
model = rf.fit(train_vec)
train_time = time.time() - t0
print(f"\n  [OK] Training complete in {train_time:.1f}s  ({train_time/60:.1f} minutes)")

# -------------------------------------------------------------
# STEP 6 - Generate Predictions on Test Set
# -------------------------------------------------------------
print(f"\n[Step 6] Generating predictions on test set ...")
predictions = model.transform(test_vec)
predictions.cache()

# -------------------------------------------------------------
# STEP 7 - Evaluate Model
# -------------------------------------------------------------
print(f"\n[Step 7] Evaluating model ...")

# AUC-ROC
auc_evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc_roc = auc_evaluator.evaluate(predictions)

# AUC-PR
auc_pr_evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)
auc_pr = auc_pr_evaluator.evaluate(predictions)

# F1, Precision, Recall, Accuracy
mc_evaluator = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
)
# PySpark 4.x: create separate evaluator instances for each metric
f1_eval        = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="f1")
acc_eval       = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy")
prec_eval      = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedPrecision")
rec_eval       = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedRecall")

f1        = f1_eval.evaluate(predictions)
accuracy  = acc_eval.evaluate(predictions)
precision = prec_eval.evaluate(predictions)
recall    = rec_eval.evaluate(predictions)

print("\n" + "=" * 60)
print("  MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"""
  Accuracy    : {accuracy:.4f}  (NOTE: misleading metric for imbalanced data!)
  AUC-ROC     : {auc_roc:.4f}  <-- PRIMARY metric (1.0 = perfect classifier)
  AUC-PR      : {auc_pr:.4f}   <-- precision-recall tradeoff
  F1 Score    : {f1:.4f}       <-- harmonic mean of precision & recall
  Precision   : {precision:.4f}    <-- of ALL predicted fraud, how many are real fraud?
  Recall      : {recall:.4f}    <-- of ALL actual fraud, how many did we catch?
""")

# -------------------------------------------------------------
# STEP 8 - Confusion Matrix
# -------------------------------------------------------------
print("[Step 8] Confusion Matrix ...")

tp = predictions.filter((fcol("prediction") == 1) & (fcol(LABEL_COL) == 1)).count()
tn = predictions.filter((fcol("prediction") == 0) & (fcol(LABEL_COL) == 0)).count()
fp = predictions.filter((fcol("prediction") == 1) & (fcol(LABEL_COL) == 0)).count()
fn = predictions.filter((fcol("prediction") == 0) & (fcol(LABEL_COL) == 1)).count()

# Avoid division by zero
fraud_catch_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

print(f"""
  Confusion Matrix:

                    Predicted:Legit  Predicted:Fraud
  Actual: Legit     {tn:<16} {fp:<16}  ({fp} false alarms)
  Actual: Fraud     {fn:<16} {tp:<16}  ({fn} missed frauds -- keep this LOW!)

  True  Positives (caught fraud)  : {tp:,}
  True  Negatives (correct legit) : {tn:,}
  False Positives (false alarm)   : {fp:,}
  False Negatives (missed fraud)  : {fn:,}  <-- most dangerous error

  Fraud catch rate  : {fraud_catch_rate:.2f}%  ({tp} of {tp+fn} actual frauds caught)
  False alarm rate  : {false_alarm_rate:.4f}%  ({fp} legit transactions wrongly flagged)
""")

# -------------------------------------------------------------
# STEP 9 - Feature Importances
# -------------------------------------------------------------
print("[Step 9] Feature Importances (ranked by importance) ...")
importances = model.featureImportances.toArray()
ranked = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)

print(f"\n  {'Rank':<6} {'Feature':<22} {'Importance':<12} Bar Chart")
print(f"  {'=' * 60}")
for rank, (feat, imp) in enumerate(ranked, 1):
    bar = "|" * max(1, int(imp * 60))
    print(f"  {rank:<6} {feat:<22} {imp:<12.4f} {bar}")

# -------------------------------------------------------------
# STEP 10 - Save Model
# -------------------------------------------------------------
print(f"\n[Step 10] Saving model to ./{MODEL_PATH}/ ...")
model.write().overwrite().save(MODEL_PATH)
print(f"  [OK] Model saved to: {MODEL_PATH}/")
print(f"\n  To reload later:")
print(f"    from pyspark.ml.classification import RandomForestClassificationModel")
print(f"    model = RandomForestClassificationModel.load('{MODEL_PATH}')")

# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("  TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 60)
print(f"""
  Train set       : {train_count:,} rows
  Test  set       : {test_count:,} rows
  Training time   : {train_time:.1f}s ({train_time/60:.1f} min)
  Model           : RandomForestClassifier (100 trees, depth 10)

  AUC-ROC         : {auc_roc:.4f}   (target: > 0.95)
  F1 Score        : {f1:.4f}        (target: > 0.80)
  Recall          : {recall:.4f}    (fraud catch rate)
  Fraud caught    : {fraud_catch_rate:.2f}%

  Model saved     : ./{MODEL_PATH}/

  Next step: Run spark_consumer_ml.py to deploy into the live pipeline.
""")
print("=" * 60)

spark.stop()
