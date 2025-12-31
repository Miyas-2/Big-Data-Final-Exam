import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp, when
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import StructType, DoubleType, IntegerType, StringType
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel

# ======================================================
# 1. Spark Session (LOCAL FS + Cassandra)
# ======================================================
spark = SparkSession.builder \
    .appName("HealthcareRiskStreamingCassandra") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .config("spark.cassandra.connection.host", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✓ Spark Streaming Started")

# ======================================================
# 2. Load Model & SCALER (Path Terbaru Anda)
# ======================================================
model_root = "healthcare_risk_model_20251231_042343"
model_path = os.path.abspath(f"{model_root}/model/gbt_model")
scaler_path = os.path.abspath(f"{model_root}/scaler")

model = GBTClassificationModel.load(model_path)
scaler = StandardScalerModel.load(scaler_path)
print("✓ GBT Model and Scaler loaded successfully")

# ======================================================
# 3. Kafka JSON Schema
# ======================================================
schema = StructType() \
    .add("patient_id", IntegerType()) \
    .add("Heart Rate", DoubleType()) \
    .add("Respiratory Rate", DoubleType()) \
    .add("Body Temperature", DoubleType()) \
    .add("Oxygen Saturation", DoubleType()) \
    .add("Systolic Blood Pressure", DoubleType()) \
    .add("Diastolic Blood Pressure", DoubleType()) \
    .add("Age", IntegerType()) \
    .add("Gender", StringType()) \
    .add("Weight (kg)", DoubleType()) \
    .add("Height (m)", DoubleType()) \
    .add("Derived_HRV", DoubleType()) \
    .add("Derived_BMI", DoubleType()) \
    .add("Derived_MAP", DoubleType()) \
    .add("Derived_Pulse_Pressure", DoubleType())

# ======================================================
# 4. Read Kafka Stream
# ======================================================
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "healthcare-vitals") \
    .option("startingOffsets", "latest") \
    .load()

# Parsing data dan menambahkan Gender_encoded (asumsi Male:1, Female:0)
parsed_df = kafka_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*") \
 .withColumn("Gender_encoded", when(col("Gender") == "Male", 1.0).otherwise(0.0))

# ======================================================
# 5. Feature Engineering (SYNC 12 FEATURES)
# ======================================================
feature_cols = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Derived_BMI', 
    'Age', 'Gender_encoded', 'Derived_HRV', 'Derived_MAP', 'Derived_Pulse_Pressure'
]

# Step A: Assemble ke raw_features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df_assembled = assembler.transform(parsed_df)

# Step B: Scaling menggunakan Scaler hasil training
df_scaled = scaler.transform(df_assembled)

# ======================================================
# 6. Model Prediction
# ======================================================
predictions = model.transform(df_scaled)

# ======================================================
# 7. Prepare Output for Cassandra
# ======================================================
output_df = predictions \
    .withColumn("event_time", current_timestamp()) \
    .withColumn("prob_array", vector_to_array(col("probability"))) \
    .select(
        col("patient_id"),
        col("event_time"),
        col("Heart Rate").alias("heart_rate"),
        col("Respiratory Rate").alias("respiratory_rate"),
        col("Body Temperature").alias("body_temperature"),
        col("Oxygen Saturation").alias("oxygen_saturation"),
        col("Systolic Blood Pressure").alias("systolic_bp"),
        col("Diastolic Blood Pressure").alias("diastolic_bp"),
        col("Age").alias("age"),
        col("Gender").alias("gender"),
        col("Weight (kg)").alias("weight_kg"),      # <--- TAMBAHKAN INI
        col("Height (m)").alias("height_m"),       # <--- TAMBAHKAN INI
        col("Derived_HRV").alias("derived_hrv"),
        col("Derived_Pulse_Pressure").alias("derived_pulse_pressure"),
        col("Derived_BMI").alias("derived_bmi"),
        col("Derived_MAP").alias("derived_map"),
        col("prediction").cast("int"),              # Cast ke int agar sesuai schema Cassandra
        col("prob_array")[1].alias("probability_high_risk")
    )
# ======================================================
# 8. Write Streaming to Cassandra
# ======================================================
def write_to_cassandra(batch_df, batch_id):
    batch_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(keyspace="healthcare", table="vital_predictions") \
        .save()

query = output_df.writeStream \
    .foreachBatch(write_to_cassandra) \
    .outputMode("append") \
    .start()

print("✓ Streaming process is running and saving to Cassandra...")
query.awaitTermination()