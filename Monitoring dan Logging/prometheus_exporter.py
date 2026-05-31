from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import random
import sys
import os

# Tambahkan path ke folder Membangun_model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Membangun model'))

import joblib
import numpy as np

# ============================================
# METRICS
# ============================================
# Counter untuk jumlah prediksi
prediction_count = Counter('prediction_total', 'Total number of predictions made')
error_count = Counter('prediction_errors', 'Total number of prediction errors')

# Histogram untuk latency prediksi
prediction_latency = Histogram('prediction_latency_seconds', 'Time taken for prediction')

# Gauge untuk resource usage
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')

# Counter untuk nilai prediksi per kategori
price_category_count = Counter('price_category_predictions', 'Predictions by price category', ['category'])

# ============================================
# LOAD MODEL
# ============================================
try:
    model = joblib.load('../Membangun model/best_model.pkl')
    scaler = joblib.load('../Membangun model/scaler.pkl')
    print("Model dan scaler berhasil dimuat.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# ============================================
# SIMULASI PREDIKSI
# ============================================
def predict(features):
    """Simulate model prediction"""
    if model is None:
        return None
    
    try:
        # Record latency
        with prediction_latency.time():
            # Scale features
            features_scaled = scaler.transform([features])
            # Predict
            prediction = model.predict(features_scaled)[0]
        
        # Increment counters
        prediction_count.inc()
        
        # Categorize prediction
        if prediction < 1:
            price_category_count.labels(category='Very Low').inc()
        elif prediction < 2:
            price_category_count.labels(category='Low').inc()
        elif prediction < 3:
            price_category_count.labels(category='Medium').inc()
        elif prediction < 4:
            price_category_count.labels(category='High').inc()
        else:
            price_category_count.labels(category='Very High').inc()
        
        return prediction
        
    except Exception as e:
        error_count.inc()
        print(f"Prediction error: {e}")
        return None

# ============================================
# SIMULASI RESOURCE MONITORING
# ============================================
def update_resource_metrics():
    """Update system resource metrics"""
    try:
        import psutil
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
    except ImportError:
        # Fallback jika psutil tidak tersedia, gunakan nilai acak
        cpu_usage.set(random.uniform(10, 80))
        memory_usage.set(random.uniform(20, 70))

# ============================================
# MAIN FUNCTION
# ============================================
if __name__ == '__main__':
    # Menjalankan Prometheus metrics di port 8000
    start_http_server(8000)
    print("Prometheus berjalan di http://localhost:8000/metrics")
    
    # Simulasi prediksi
    sample_features = [3.0, 30.0, 5.0, 1.0, 1000.0, 2.5, 34.0, -118.0, 2.0, 0.2, 400.0]
    
    while True:
        # Simulate prediction every 10 seconds
        prediction = predict(sample_features)
        if prediction:
            print(f"Prediction: {prediction:.4f}")
        
        # Update resource metrics
        update_resource_metrics()
        
        time.sleep(10)