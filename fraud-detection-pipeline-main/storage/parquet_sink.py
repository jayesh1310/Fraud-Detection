"""
parquet_sink.py
---------------
Utility helpers for persisting processed transaction DataFrames to
Parquet on local disk (or HDFS/S3 — just swap the OUTPUT_BASE_PATH).
"""

import os
from datetime import datetime

from pyspark.sql import DataFrame


OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH", "/data/output")


def write_batch(df: DataFrame, table_name: str, partition_cols: list[str] | None = None) -> None:
    """
    Persist a batch DataFrame to Parquet.

    Args:
        df:              PySpark DataFrame to write.
        table_name:      Sub-directory name under OUTPUT_BASE_PATH.
        partition_cols:  Optional list of columns to partition by.
    """
    path = os.path.join(OUTPUT_BASE_PATH, table_name)
    writer = df.write.mode("append").format("parquet")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save(path)
    print(f"[Storage] Wrote batch to {path}")


def write_stream(df: DataFrame, table_name: str, checkpoint_suffix: str = "") -> None:
    """
    Start a Structured Streaming write to Parquet.

    Args:
        df:                PySpark streaming DataFrame.
        table_name:        Sub-directory name under OUTPUT_BASE_PATH.
        checkpoint_suffix: Optional suffix appended to the checkpoint directory name.
    """
    output_path = os.path.join(OUTPUT_BASE_PATH, table_name)
    checkpoint_path = os.path.join(
        OUTPUT_BASE_PATH,
        "checkpoints",
        f"{table_name}{checkpoint_suffix}_{datetime.utcnow().strftime('%Y%m%d')}",
    )

    query = (
        df.writeStream.outputMode("append")
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime="10 seconds")
        .start()
    )
    print(f"[Storage] Streaming sink started → {output_path}")
    return query
