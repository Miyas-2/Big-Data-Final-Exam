import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.functions import vector_to_array

# ======================================================
# 1. SPARK SESSION & LOGGING
# ======================================================
spark = SparkSession.builder \
    .appName("ThresholdTesting") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ======================================================
# 2. LOAD MODEL & SCALER (PASTIKAN FOLDERNYA BENAR)
# ======================================================
# Gunakan nama folder yang Anda buat saat menyimpan tadi
model_root = "healthcare_risk_model_20251231_042343" 
model_path = os.path.abspath(f"{model_root}/model/gbt_model")
scaler_path = os.path.abspath(f"{model_root}/scaler")

try:
    model = GBTClassificationModel.load(model_path)
    scaler = StandardScalerModel.load(scaler_path)
    print("✓ Model and Scaler loaded successfully (12 Features Mode)")
except Exception as e:
    print(f"✗ ERROR: Gagal memuat model/scaler: {e}")
    sys.exit(1)

# ======================================================
# 3. GENERATE SIMULASI DATA (12 FITUR)
# ======================================================
test_data = []
for hr in range(40, 111):
    test_data.append({
        "Heart Rate": float(hr),
        "Respiratory Rate": 16.0,
        "Body Temperature": 36.6,
        "Oxygen Saturation": 98.0,
        "Systolic Blood Pressure": 120.0,
        "Diastolic Blood Pressure": 80.0,
        "Derived_BMI": 22.5,  # Ideal
        "Age": 30,
        "Gender_encoded": 1,
        "Derived_HRV": 55.0,
        "Derived_MAP": 93.3,
        "Derived_Pulse_Pressure": 40.0
    })

df_test = spark.createDataFrame(test_data)

# ======================================================
# 4. FEATURE SELECTION (HARUS PERSIS SAMA DENGAN TRAINING)
# ======================================================
# Urutan ini HARUS sama dengan list 'feature_columns' di notebook Anda
feature_cols = [
    'Heart Rate',
    'Respiratory Rate',
    'Body Temperature',
    'Oxygen Saturation',
    'Systolic Blood Pressure',
    'Diastolic Blood Pressure',
    'Derived_BMI',
    'Age',
    'Gender_encoded',
    'Derived_HRV',
    'Derived_MAP',
    'Derived_Pulse_Pressure'
]

# Total harus 12
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df_vector = assembler.transform(df_test)

# ======================================================
# 5. TRANSFORM & PREDICT
# ======================================================
try:
    # Scaling data mentah menjadi 'features'
    df_scaled = scaler.transform(df_vector)
    
    # Prediksi menggunakan model GBT
    predictions = model.transform(df_scaled)
    
    # Ambil probabilitas High Risk
    results = predictions.withColumn("prob_high", vector_to_array("probability")[1]) \
        .select("Heart Rate", "prediction", "prob_high") \
        .orderBy("Heart Rate")

    # ======================================================
    # 6. OUTPUT AKHIR (PANDAS)
    # ======================================================
    print("\n" + "="*60)
    print("HASIL ANALISIS THRESHOLD (12 FEATURES SYNC)")
    print("="*60)
    
    final_pdf = results.toPandas()
    # Tampilkan rentang normal HR
    print(final_pdf[(final_pdf['Heart Rate'] >= 55) & (final_pdf['Heart Rate'] <= 95)].to_string(index=False))
    
    print("="*60)
    low_risk_area = final_pdf[final_pdf['prediction'] == 0.0]
    if not low_risk_area.empty:
        print(f"✓ MODEL NORMAL PADA HR: {low_risk_area['Heart Rate'].min()} - {low_risk_area['Heart Rate'].max()}")
    else:
        print("✗ MASIH HIGH RISK: Cek apakah nilai BP/BMI di simulasi sudah masuk range normal.")
    print("="*60 + "\n")

except Exception as e:
    print(f"✗ RUNTIME ERROR: {e}")