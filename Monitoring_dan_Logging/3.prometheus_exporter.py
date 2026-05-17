"""
prometheus_exporter.py
========================
Custom Prometheus exporter untuk monitoring tambahan
di luar metrik yang sudah ada di inference.py.

Author: Devin Novansyah

Cara menjalankan:
    pip install prometheus-client psutil requests
    python prometheus_exporter.py

Metrics yang diekspos (port 8000):
    - titanic_model_accuracy_gauge
    - titanic_system_cpu_usage_percent
    - titanic_system_memory_usage_percent
    - titanic_api_up
    - titanic_total_predictions_served
"""

import time
import psutil
import requests
from prometheus_client import (
    start_http_server,
    Gauge, Counter, Info
)

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
EXPORTER_PORT = 8000
API_URL       = 'http://localhost:5001'
SCRAPE_INTERVAL = 15   # detik

# ──────────────────────────────────────────────
# CUSTOM METRICS
# ──────────────────────────────────────────────

# 1. Model accuracy (dari hasil training terakhir)
MODEL_ACCURACY = Gauge(
    'titanic_model_accuracy_gauge',
    'Akurasi model Titanic hasil training terakhir'
)

# 2. CPU usage sistem
CPU_USAGE = Gauge(
    'titanic_system_cpu_usage_percent',
    'Persentase penggunaan CPU sistem'
)

# 3. Memory usage sistem
MEMORY_USAGE = Gauge(
    'titanic_system_memory_usage_percent',
    'Persentase penggunaan memori sistem'
)

# 4. Status API (up/down)
API_UP = Gauge(
    'titanic_api_up',
    'Status API inference (1=up, 0=down)'
)

# 5. Total prediksi yang sudah dilayani
TOTAL_PREDICTIONS = Gauge(
    'titanic_total_predictions_served',
    'Total prediksi yang sudah dilayani sejak server start'
)

# 6. Info model
MODEL_INFO = Info(
    'titanic_model',
    'Informasi model yang sedang digunakan'
)


# ──────────────────────────────────────────────
# FUNGSI PENGUMPUL METRIK
# ──────────────────────────────────────────────

def collect_system_metrics():
    """Kumpulkan metrik sistem (CPU & Memory)."""
    cpu   = psutil.cpu_percent(interval=1)
    mem   = psutil.virtual_memory().percent
    CPU_USAGE.set(cpu)
    MEMORY_USAGE.set(mem)
    print(f'[SYS] CPU: {cpu:.1f}% | Memory: {mem:.1f}%')


def collect_api_metrics():
    """Cek status API dan kumpulkan metrik dari /metrics endpoint."""
    try:
        resp = requests.get(f'{API_URL}/health', timeout=5)
        if resp.status_code == 200:
            API_UP.set(1)
            print(f'[API] Status: UP ✅')
        else:
            API_UP.set(0)
            print(f'[API] Status: DOWN ❌ (HTTP {resp.status_code})')
    except Exception as e:
        API_UP.set(0)
        print(f'[API] Status: DOWN ❌ ({e})')


def set_model_info(accuracy: float = 0.82):
    """Set informasi model (akurasi dari hasil training)."""
    MODEL_ACCURACY.set(accuracy)
    MODEL_INFO.info({
        'name'   : 'RandomForestClassifier',
        'dataset': 'Titanic',
        'author' : 'Devin Novansyah',
        'version': '1.0.0'
    })
    print(f'[MODEL] Accuracy: {accuracy:.4f}')


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  PROMETHEUS EXPORTER - Titanic')
    print('  Author: Devin Novansyah')
    print(f'  Exporter port: {EXPORTER_PORT}')
    print('=' * 55)

    # Start HTTP server untuk Prometheus scrape
    start_http_server(EXPORTER_PORT)
    print(f'✅ Exporter berjalan di http://localhost:{EXPORTER_PORT}/metrics\n')

    # Set model info sekali di awal
    # Ganti nilai accuracy dengan hasil aktual dari modelling_tuning.py
    set_model_info(accuracy=0.8268)

    total_pred = 0

    while True:
        try:
            collect_system_metrics()
            collect_api_metrics()

            # Simulasi counter prediksi (ganti dengan query ke API nyata)
            TOTAL_PREDICTIONS.set(total_pred)
            total_pred += 1

            print(f'[TICK] Metrik diperbarui. Next update in {SCRAPE_INTERVAL}s...\n')
        except Exception as e:
            print(f'[ERROR] {e}')

        time.sleep(SCRAPE_INTERVAL)


if __name__ == '__main__':
    main()
