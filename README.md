# Membangun Sistem Machine Learning - Devin Novansyah

Repositori ini berisi submission proyek akhir **Membangun Sistem Machine Learning** di Dicoding Indonesia.

## 👤 Identitas

| Info | Detail |
|------|--------|
| **Nama** | Devin Novansyah |
| **Username Dicoding** | devin-novansyah16 |
| **Dataset** | Titanic - Machine Learning from Disaster |
| **Task** | Binary Classification (Survived / Not Survived) |

---

## 📁 Struktur Repository

```
Membangun-Sistem-Machine-Learning-Dicoding/
├── Eksperimen_SML_Devin-Novansyah.txt   ← Link repo Kriteria 1
├── Workflow-CI.txt                        ← Link repo Kriteria 3
├── Membangun_model/                       ← Kriteria 2
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── titanic_preprocessing/
│   │   └── train_preprocessed.csv
│   ├── screenshoot_dashboard.png
│   ├── screenshoot_artifak.png
│   └── requirements.txt
└── Monitoring_dan_Logging/                ← Kriteria 4
    ├── 1.bukti_serving/
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.bukti_monitoring_Prometheus/
    ├── 5.bukti_monitoring_Grafana/
    ├── 6.bukti_alerting_Grafana/
    └── 7.inference.py
```

---

## 📋 Kriteria Submission

### Kriteria 1 — Eksperimen Dataset
🔗 Repository: [Eksperimen_SML_Devin-Novansyah](https://github.com/devin-novansyah16/Eksperimen_SML_Devin-Novansyah)

- ✅ Data loading, EDA, dan preprocessing pada notebook
- ✅ Script otomatisasi `automate_Devin-Novansyah.py`
- ✅ GitHub Actions workflow untuk preprocessing otomatis

**Tahapan Preprocessing:**
| No | Tahap | Keterangan |
|----|-------|------------|
| 1 | Seleksi Fitur | Drop: PassengerId, Name, Ticket, Cabin |
| 2 | Handling Missing | Age → median, Embarked → modus |
| 3 | Feature Engineering | FamilySize, IsAlone, AgeGroup |
| 4 | Encoding | Sex (Label), Embarked & AgeGroup (One-Hot) |
| 5 | Scaling | Age, Fare, FamilySize → StandardScaler |

---

### Kriteria 2 — Membangun Model ML
📂 Folder: `Membangun_model/`

- ✅ `modelling.py` → MLflow autolog tanpa hyperparameter tuning
- ✅ `modelling_tuning.py` → Manual logging + GridSearchCV tuning
- ✅ Screenshot MLflow dashboard dan artefak

**Model:** Random Forest Classifier
**Metrik yang di-log:**

| Metrik | Nilai |
|--------|-------|
| Accuracy | ~0.80 |
| Precision | ~0.77 |
| Recall | ~0.71 |
| F1-Score | ~0.74 |
| ROC-AUC | ~0.84 |
| CV Mean Accuracy | ~0.83 |

---

### Kriteria 3 — Workflow CI
🔗 Repository: [Workflow-CI](https://github.com/devin-novansyah16/Workflow-CI)

- ✅ Folder MLProject dengan `modelling.py`, `conda.yaml`, `MLProject`
- ✅ GitHub Actions CI menggunakan `mlflow run`
- ✅ Artefak tersimpan otomatis ke repository setiap trigger

---

### Kriteria 4 — Monitoring & Logging
📂 Folder: `Monitoring_dan_Logging/`

- ✅ Serving model via FastAPI (`7.inference.py`)
- ✅ Monitoring Prometheus dengan 5+ metrik
- ✅ Dashboard Grafana dengan 5 panel
- ✅ Alerting Grafana (High Latency Alert)

**Metrik yang dimonitor:**
| No | Metrik | Keterangan |
|----|--------|------------|
| 1 | `titanic_request_total` | Total request prediksi |
| 2 | `titanic_request_latency_seconds` | Latensi request |
| 3 | `titanic_prediction_class_total` | Prediksi per kelas |
| 4 | `titanic_prediction_probability` | Distribusi probabilitas |
| 5 | `titanic_active_requests` | Request aktif |
| 6 | `titanic_system_cpu_usage_percent` | CPU usage |
| 7 | `titanic_system_memory_usage_percent` | Memory usage |

---

## 🛠️ Tech Stack

| Tools | Kegunaan |
|-------|----------|
| Python 3.10 | Bahasa pemrograman utama |
| Scikit-Learn | Model machine learning |
| MLflow | Experiment tracking & model registry |
| FastAPI | Model serving |
| Prometheus | Monitoring metrics |
| Grafana | Visualisasi & alerting |
| Docker | Container untuk Prometheus & Grafana |
| GitHub Actions | CI/CD workflow |
