FROM apache/spark:3.5.0

# Build context is the PROJECT ROOT (set in docker-compose.yml) so this
# Dockerfile can reach both consumer/ and features/ in a single build.
#
# The official apache/spark image runs as UID 185 (spark user).
# Switch to root to install packages and download JARs, then drop back.
USER root

# Install Python 3 and pip (the base image is minimal — no pip by default)
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Path is relative to build context (project root), not this file's directory
COPY consumer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Pre-download Kafka connector JARs into /opt/spark/jars/ at build time.
# Avoids runtime Ivy resolution which writes to /home/spark/.ivy2/cache and
# fails when running as the non-root spark user.
# ---------------------------------------------------------------------------
ENV MAVEN_REPO=https://repo1.maven.org/maven2
ENV SPARK_JARS_DIR=/opt/spark/jars

# Delete ALL existing Kafka-related JARs from the base image before downloading
# the exact versions needed for Spark 3.5.0 + Scala 2.12.
# Having two versions of the same JAR on the classpath causes:
#   ClassCastException: DataSourceRDDPartition cannot be cast to DataSourceRDDPartition
RUN rm -f "${SPARK_JARS_DIR}"/commons-pool2-*.jar \
          "${SPARK_JARS_DIR}"/kafka-clients-*.jar \
          "${SPARK_JARS_DIR}"/kafka-*.jar \
          "${SPARK_JARS_DIR}"/spark-sql-kafka-*.jar \
          "${SPARK_JARS_DIR}"/spark-token-provider-kafka-*.jar \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar" \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar" \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar" \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.0/spark-token-provider-kafka-0-10_2.12-3.5.0.jar" \
    && echo "--- Kafka JARs installed in ${SPARK_JARS_DIR} ---" \
    && ls -lh "${SPARK_JARS_DIR}"/*.jar | grep -E "kafka|commons-pool2"

WORKDIR /app

# Bug 1 fix: copy the features/ package from the project root so Python can
# import feature_engineering at runtime on both driver and executor.
COPY features/ /app/features/

# Copy consumer application source
COPY consumer/ /app/

# Bug 1 fix: add /app to PYTHONPATH so `from features.feature_engineering import ...`
# resolves correctly when Spark spawns Python worker subprocesses on the executor.
ENV PYTHONPATH=/app

# Drop back to the non-root spark user for the actual process
USER spark

# --packages flag removed: JARs are already on the Spark classpath via /opt/spark/jars/
# --master flag removed: spark-submit reads it from the SPARK_MASTER env var or compose
CMD ["spark-submit", \
     "--master", "local[2]", \
     "spark_consumer_ml.py"]
