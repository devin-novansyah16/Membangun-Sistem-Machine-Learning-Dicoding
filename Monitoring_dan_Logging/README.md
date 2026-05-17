# Monitoring dan Logging - Devin Novansyah

Folder ini berisi semua file untuk monitoring & logging sistem ML Titanic
menggunakan **Prometheus** + **Grafana**.

## Struktur Folder

```
Monitoring dan Logging/
├── 1.bukti_serving/              ← Screenshot bukti serving model (isi sendiri)
├── 2.prometheus.yml              ← Konfigurasi Prometheus
├── 3.prometheus_exporter.py      ← Custom Prometheus exporter
├── 4.bukti monitoring Prometheus/ ← Screenshot Prometheus (isi sendiri)
│   ├── 1.monitoring_request_total.png
│   ├── 2.monitoring_latency.png
│   ├── 3.monitoring_prediction_class.png
│   ├── 4.monitoring_prediction_probability.png
│   └── 5.monitoring_active_requests.png
├── 5.bukti monitoring Grafana/   ← Screenshot Grafana (isi sendiri)
│   ├── 1.monitoring_request_total.png
│   ├── 2.monitoring_latency.png
│   ├── 3.monitoring_prediction_class.png
│   ├── 4.monitoring_cpu_usage.png
│   └── 5.monitoring_memory_usage.png
├── 6.bukti alerting Grafana/     ← Screenshot alerting (isi sendiri)
│   ├── 1.rules_high_latency.png
│   └── 2.notifikasi_high_latency.png
├── 7.inference.py                ← FastAPI serving + Prometheus metrics
├── docker-compose.yml            ← Setup Prometheus + Grafana
└── README.md
```

---

## Cara Setup Lengkap

### Step 1 — Install dependencies
```bash
pip install fastapi uvicorn prometheus-client mlflow scikit-learn pandas psutil requests
```

### Step 2 — Serving model (bukti_serving)
```bash
# Pastikan MLflow artifacts sudah ada (dari Kriteria 2/3)
# Jalankan inference API
python 7.inference.py
# API berjalan di: http://localhost:5001
# Docs: http://localhost:5001/docs
```

### Step 3 — Jalankan Prometheus + Grafana
```bash
docker-compose up -d
# Prometheus : http://localhost:9090
# Grafana    : http://localhost:3000 (admin/admin)
```

### Step 4 — Jalankan Custom Exporter
```bash
python 3.prometheus_exporter.py
# Exporter berjalan di: http://localhost:8000/metrics
```

### Step 5 — Test prediksi
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 3, "Sex": 1, "Age": 22.0,
    "SibSp": 1, "Parch": 0, "Fare": 7.25,
    "FamilySize": 2, "IsAlone": 0
  }'
```

---

## Metrik yang Dimonitor (5 untuk Skilled)

| No | Nama Metrik | Keterangan | Tools |
|----|-------------|------------|-------|
| 1 | `titanic_request_total` | Total request prediksi | Prometheus + Grafana |
| 2 | `titanic_request_latency_seconds` | Latensi request (histogram) | Prometheus + Grafana |
| 3 | `titanic_prediction_class_total` | Jumlah prediksi per kelas | Prometheus + Grafana |
| 4 | `titanic_prediction_probability` | Distribusi probabilitas | Prometheus + Grafana |
| 5 | `titanic_active_requests` | Request aktif saat ini | Prometheus + Grafana |
| 6 | `titanic_system_cpu_usage_percent` | CPU usage sistem | Exporter + Grafana |
| 7 | `titanic_system_memory_usage_percent` | Memory usage sistem | Exporter + Grafana |
| 8 | `titanic_api_up` | Status API (up/down) | Exporter + Grafana |
| 9 | `titanic_model_accuracy_gauge` | Akurasi model | Exporter + Grafana |
| 10| `titanic_error_total` | Total error | Prometheus + Grafana |

---

## Setup Grafana Dashboard

### Tambahkan Data Source
1. Buka Grafana → **Configuration** → **Data Sources**
2. Klik **Add data source** → pilih **Prometheus**
3. URL: `http://prometheus:9090`
4. Klik **Save & Test**

### Buat Panel (5 metrik untuk Skilled)
Buat dashboard baru dengan 5 panel berikut:

| Panel | Query PromQL |
|-------|-------------|
| Request Total | `rate(titanic_request_total[5m])` |
| Latency P95 | `histogram_quantile(0.95, rate(titanic_request_latency_seconds_bucket[5m]))` |
| Prediction per Class | `titanic_prediction_class_total` |
| Prediction Probability | `histogram_quantile(0.5, rate(titanic_prediction_probability_bucket[5m]))` |
| Active Requests | `titanic_active_requests` |

**Penting:** Beri nama dashboard dengan **username Dicoding** kamu!

---

## Setup Alerting Grafana (1 alert untuk Skilled)

### Alert: High Latency
1. Buka panel **Latency P95**
2. Klik **Edit** → tab **Alert**
3. Klik **Create alert rule**
4. Konfigurasi:
   - **Name:** `High Latency Alert - Devin Novansyah`
   - **Condition:** `WHEN last() OF query IS ABOVE 1.0`
   - **For:** `1m`
   - **Message:** `Latency prediksi melebihi 1 detik!`
5. Tambahkan notification channel (Email/Slack/Webhook)
6. Klik **Save**

---

## Author
**Devin Novansyah** - Machine Learning Submission
