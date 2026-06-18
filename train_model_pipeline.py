import os
import sys
import time

os.environ["HADOOP_HOME"]       = r"C:\hadoop"
os.environ["hadoop.home.dir"]   = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col as fcol, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

TRAIN_PATH = os.path.abspath("./training_data")
MODEL_PATH = "fraud_rf_model"
LABEL_COL  = "label"

def main():
    print("=" * 60)
    print("  TASK 3 - PySpark MLlib Pipeline Training (Live Data)")
    print("=" * 60)

    spark = (
        SparkSession.builder
        .appName("FraudDetection_RF_Pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.io.native.lib.available", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n[Step 1] Loading {TRAIN_PATH} ...")
    t0 = time.time()
    df = spark.read.parquet(TRAIN_PATH)
    
    # Schema alignment based on SCHEMA.md section 4
    df = df.withColumn("label", fcol("is_fraud").cast("int")) \
           .withColumn("is_night_int", fcol("is_night_transaction").cast("int")) \
           .withColumn("lat", fcol("location.lat")) \
           .withColumn("lon", fcol("location.lon"))
    
    # Calculate class weights for balancing
    total_count = df.count()
    fraud_count = df.filter(fcol("label") == 1).count()
    legit_count = total_count - fraud_count
    
    print(f"  Total rows : {total_count:,}  (took {time.time()-t0:.1f}s)")
    print(f"  Legit (0)  : {legit_count:,} ({legit_count/total_count*100:.2f}%)")
    print(f"  Fraud (1)  : {fraud_count:,} ({fraud_count/total_count*100:.2f}%)")

    if fraud_count == 0:
        print("\nERROR: No fraud cases in this snapshot! Cannot train. Run producer longer.")
        sys.exit(1)

    weight_legit = total_count / (2.0 * legit_count)
    weight_fraud = total_count / (2.0 * fraud_count)

    df = df.withColumn("classWeight", 
        when(fcol("label") == 1, weight_fraud).otherwise(weight_legit)
    )

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()

    print(f"\n[Step 2] Building Pipeline ...")
    
    cat_indexer  = StringIndexer(inputCol="merchant_category", outputCol="cat_idx", handleInvalid="keep")
    cat_encoder  = OneHotEncoder(inputCol="cat_idx", outputCol="cat_vec")

    bucket_indexer = StringIndexer(
        inputCol="amount_bucket", outputCol="bucket_idx",
        stringOrderType="alphabetAsc", handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=[
            "amount", "hour_of_day", "is_night_int",
            "geo_risk_score", "lat", "lon",
            "cat_vec", "bucket_idx"
        ],
        outputCol="raw_features",
        handleInvalid="skip"
    )

    scaler = StandardScaler(inputCol="raw_features", outputCol="features")

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=[cat_indexer, cat_encoder, bucket_indexer, assembler, scaler, rf])

    print("\n[Step 3] Training Pipeline ... (this takes a minute)")
    t1 = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - t1
    print(f"  [OK] Training complete in {train_time:.1f}s")

    print("\n[Step 4] Evaluating Model ...")
    predictions = model.transform(test_df)

    auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc_roc = auc_eval.evaluate(predictions)
    
    auc_pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    auc_pr = auc_pr_eval.evaluate(predictions)

    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = f1_eval.evaluate(predictions)

    tp = predictions.filter((fcol("prediction") == 1) & (fcol("label") == 1)).count()
    tn = predictions.filter((fcol("prediction") == 0) & (fcol("label") == 0)).count()
    fp = predictions.filter((fcol("prediction") == 1) & (fcol("label") == 0)).count()
    fn = predictions.filter((fcol("prediction") == 0) & (fcol("label") == 1)).count()

    fraud_catch_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  AUC-ROC     : {auc_roc:.4f}")
    print(f"  AUC-PR      : {auc_pr:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    
    print("\n  Confusion Matrix:")
    print(f"                    Predicted:Legit  Predicted:Fraud")
    print(f"  Actual: Legit     {tn:<16} {fp:<16}  ({fp} false alarms)")
    print(f"  Actual: Fraud     {fn:<16} {tp:<16}  ({fn} missed frauds)")
    print(f"\n  Fraud catch rate: {fraud_catch_rate:.2f}%")
    print(f"  False alarm rate: {false_alarm_rate:.4f}%")

    print(f"\n[Step 5] Saving model to ./{MODEL_PATH}/ ...")
    model.write().overwrite().save(MODEL_PATH)
    print(f"  [OK] Model saved.")
    
    spark.stop()

if __name__ == "__main__":
    main()
