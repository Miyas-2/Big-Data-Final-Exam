from kafka import KafkaProducer
import json
import time
import random
import math

# ======================================================
# 1. KONFIGURASI SKENARIO PASIEN (UNTUK DEMO)
# ======================================================
PATIENTS = {
    1: {"name": "Stabil (Low Risk)", "gender": "Male", "age": 25, "weight": 70.0, "height": 1.75, "risk_type": "low"},
    2: {"name": "Kritis (High Risk)", "gender": "Female", "age": 72, "weight": 65.0, "height": 1.60, "risk_type": "high"},
    3: {"name": "Berubah-ubah (Fluktuatif)", "gender": "Male", "age": 45, "weight": 85.0, "height": 1.70, "risk_type": "volatile"},
    4: {"name": "Pemulihan", "gender": "Female", "age": 30, "weight": 55.0, "height": 1.65, "risk_type": "low"},
    5: {"name": "Hipertensi", "gender": "Male", "age": 55, "weight": 90.0, "height": 1.80, "risk_type": "high"}
}

# State awal tanda vital
vitals_state = {
    1: {"hr": 72, "spo2": 98.5, "sys": 120, "dia": 80}, # Normal
    2: {"hr": 110, "spo2": 91.0, "sys": 150, "dia": 95}, # Kritis
    3: {"hr": 80, "spo2": 96.0, "sys": 130, "dia": 85},  # Volatile base
    4: {"hr": 65, "spo2": 99.0, "sys": 115, "dia": 75},  # Normal
    5: {"hr": 85, "spo2": 95.0, "sys": 165, "dia": 105}  # High BP
}

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def get_simulated_vitals(p_id, t):
    p = PATIENTS[p_id]
    state = vitals_state[p_id]
    
    if p["risk_type"] == "low":
        # Fluktuasi sangat kecil di angka aman
        hr = state["hr"] + random.uniform(-1, 1)
        spo2 = state["spo2"] + random.uniform(-0.1, 0.1)
        sys = state["sys"] + random.uniform(-1, 1)
        
    elif p["risk_type"] == "high":
        # Angka tetap buruk (High Risk)
        hr = state["hr"] + random.uniform(-2, 2)
        spo2 = state["spo2"] + random.uniform(-0.2, 0.2)
        sys = state["sys"] + random.uniform(-3, 3)
        
    elif p["risk_type"] == "volatile":
        # Menggunakan fungsi SINUS untuk membuat gelombang naik turun
        # Pasien akan normal lalu sesak napas (high risk) setiap 30 detik
        wave = math.sin(t / 10) 
        hr = 80 + (wave * 30) # 50 s/d 110
        spo2 = 95 + (wave * -5) # 100 s/d 90
        sys = 130 + (wave * 25) # 105 s/d 155

    # Clamp values
    hr = max(40, min(180, hr))
    spo2 = max(80, min(100, spo2))
    
    # Update state untuk fluktuasi berikutnya
    vitals_state[p_id]["hr"] = hr
    vitals_state[p_id]["spo2"] = spo2
    
    return hr, spo2, sys

# ======================================================
# 2. RUN LOOP
# ======================================================
print("🚀 Demo Producer Running (Realistic Scenarios)...")
t = 0
try:
    while True:
        for p_id in PATIENTS:
            hr, spo2, sys = get_simulated_vitals(p_id, t)
            p = PATIENTS[p_id]
            
            # Hitung pendukung lainnya
            dia = vitals_state[p_id].get("dia", 80) + random.uniform(-1, 1)
            pp = sys - dia
            bmi = p["weight"] / (p["height"] ** 2)
            
            message = {
                "patient_id": p_id,
                "Heart Rate": round(hr, 2),
                "Respiratory Rate": random.uniform(12, 25) if p_id != 2 else random.uniform(25, 35),
                "Body Temperature": random.uniform(36.5, 37.5) if p_id != 2 else random.uniform(38.0, 39.5),
                "Oxygen Saturation": round(spo2, 2),
                "Systolic Blood Pressure": round(sys, 2),
                "Diastolic Blood Pressure": round(dia, 2),
                "Age": p["age"],
                "Gender": p["gender"],
                "Weight (kg)": p["weight"],
                "Height (m)": p["height"],
                "Derived_HRV": round(random.uniform(20, 100), 2),
                "Derived_Pulse_Pressure": round(pp, 2),
                "Derived_BMI": round(bmi, 2),
                "Derived_MAP": round(dia + (pp/3), 2)
            }
            producer.send("healthcare-vitals", value=message)
            
        print(f"[{time.strftime('%H:%M:%S')}] Batch sent for 5 patients (Mixed Risk Scenarios)")
        t += 1
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")