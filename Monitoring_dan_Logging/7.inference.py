"""
inference.py
=============
Script serving model Titanic menggunakan FastAPI.
Mengekspos endpoint prediksi dan metrik Prometheus.

Author  : Devin Novansyah
Dataset : Titanic - Machine Learning from Disaster

Cara menjalankan:
    pip install fastapi uvicorn prometheus-client mlflow scikit-learn pandas
    python inference.py

Endpoint:
    POST /predict       → prediksi survival
    GET  /metrics       → Prometheus metrics
    GET  /health        → health check
    GET  /docs          → Swagger UI
"""

import time
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response
import uvicorn
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
MODEL_URI    = "models:/titanic-model/latest"   # Ganti jika pakai path lokal
LOCAL_MODEL  = "./mlruns"                        # Fallback: path mlruns lokal
APP_NAME     = "Titanic Survival Predictor"
APP_VERSION  = "1.0.0"
APP_AUTHOR   = "Devin Novansyah"


# ──────────────────────────────────────────────
# PROMETHEUS METRICS (minimal 5 untuk Skilled)
# ──────────────────────────────────────────────

# 1. Total request yang masuk
REQUEST_COUNT = Counter(
    'titanic_request_total',
    'Total jumlah request prediksi yang diterima',
    ['method', 'endpoint', 'status']
)

# 2. Latensi prediksi (histogram)
REQUEST_LATENCY = Histogram(
    'titanic_request_latency_seconds',
    'Latensi request prediksi dalam detik',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# 3. Distribusi probabilitas prediksi
PREDICTION_PROBABILITY = Histogram(
    'titanic_prediction_probability',
    'Distribusi probabilitas prediksi survival (0.0 - 1.0)',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# 4. Counter prediksi per kelas
PREDICTION_CLASS_COUNT = Counter(
    'titanic_prediction_class_total',
    'Jumlah prediksi per kelas (survived/not_survived)',
    ['predicted_class']
)

# 5. Jumlah request aktif (gauge)
ACTIVE_REQUESTS = Gauge(
    'titanic_active_requests',
    'Jumlah request yang sedang diproses'
)

# 6. Error count
ERROR_COUNT = Counter(
    'titanic_error_total',
    'Total error yang terjadi',
    ['error_type']
)

# 7. Model info gauge
MODEL_INFO = Gauge(
    'titanic_model_info',
    'Informasi model yang sedang digunakan',
    ['version', 'author']
)


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_model():
    """
    Load model dengan mencari MLmodel di seluruh subfolder mlruns.
    Kompatibel dengan Windows dan Linux/Mac.
    """
    import glob, os
    from pathlib import Path

    # Direktori tempat script ini berada
    script_dir = Path(__file__).parent.resolve()

    print(f'[INFO] Script berada di: {script_dir}')

    # ── Cari semua file MLmodel secara rekursif ─
    # Cari di direktori script dan semua parent-nya
    search_roots = [
        script_dir,
        script_dir.parent,
        script_dir.parent / 'Membangun_model',
        script_dir.parent / 'Workflow-CI' / 'MLProject',
        Path.cwd(),
        Path.cwd().parent,
    ]

    print('[INFO] Mencari model di lokasi berikut:')
    all_mlmodel_paths = []
    for root in search_roots:
        if root.exists():
            print(f'       - {root}')
            found = list(root.rglob('MLmodel'))
            # Filter hanya yang ada di dalam folder bernama 'model'
            found = [p for p in found if p.parent.name in ('model', 'artifacts')]
            all_mlmodel_paths.extend(found)

    if not all_mlmodel_paths:
        raise RuntimeError(
            '\n\n'
            '❌ Tidak ada model ditemukan!\n'
            '\n'
            'Solusi:\n'
            '  1. Jalankan dulu di folder Membangun_model:\n'
            '       python modelling.py\n'
            '  2. Copy folder "mlruns" yang terbentuk ke folder ini\n'
            '  3. Lalu jalankan lagi: python 7.inference.py\n'
        )

    print(f'[INFO] Ditemukan {len(all_mlmodel_paths)} model:')
    for p in all_mlmodel_paths:
        print(f'       - {p}')

    # Ambil model terbaru berdasarkan waktu modifikasi
    latest_mlmodel = max(all_mlmodel_paths, key=lambda p: p.stat().st_mtime)
    model_dir      = str(latest_mlmodel.parent)

    print(f'[INFO] Menggunakan model terbaru: {model_dir}')

    try:
        model = mlflow.sklearn.load_model(model_dir)
        print(f'[INFO] ✅ Model berhasil dimuat!')
        return model
    except Exception as e:
        raise RuntimeError(
            f'❌ Model ditemukan tapi gagal dimuat: {e}\n'
            f'   Path: {model_dir}\n'
            f'   Pastikan scikit-learn versi yang sama dengan saat training.\n'
        )


# ──────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────
app   = FastAPI(title=APP_NAME, version=APP_VERSION, description=f'Author: {APP_AUTHOR}')
model = None   # Di-load saat startup


@app.on_event('startup')
async def startup_event():
    global model
    model = load_model()
    MODEL_INFO.labels(version=APP_VERSION, author=APP_AUTHOR).set(1)
    print(f'[INFO] 🚀 {APP_NAME} v{APP_VERSION} siap menerima request!')


# ──────────────────────────────────────────────
# SCHEMA INPUT
# ──────────────────────────────────────────────
class PassengerInput(BaseModel):
    Pclass    : int   = Field(..., ge=1, le=3, description='Kelas tiket (1/2/3)')
    Sex       : int   = Field(..., ge=0, le=1, description='0=female, 1=male')
    Age       : float = Field(..., ge=0,       description='Umur penumpang')
    SibSp     : int   = Field(..., ge=0,       description='Jumlah saudara/pasangan')
    Parch     : int   = Field(..., ge=0,       description='Jumlah orang tua/anak')
    Fare      : float = Field(..., ge=0,       description='Harga tiket')
    FamilySize: int   = Field(..., ge=1,       description='Ukuran keluarga (SibSp+Parch+1)')
    IsAlone   : int   = Field(..., ge=0, le=1, description='1 jika sendirian')

    class Config:
        json_schema_extra = {
            'example': {
                'Pclass': 3, 'Sex': 1, 'Age': 22.0,
                'SibSp': 1, 'Parch': 0, 'Fare': 7.25,
                'FamilySize': 2, 'IsAlone': 0
            }
        }


class PredictionOutput(BaseModel):
    survived         : int
    survived_label   : str
    probability_survive   : float
    probability_not_survive: float
    latency_ms       : float


# ──────────────────────────────────────────────
# ENDPOINT: /predict
# ──────────────────────────────────────────────
@app.post('/predict', response_model=PredictionOutput, tags=['Prediction'])
async def predict(passenger: PassengerInput):
    """
    Prediksi kemungkinan selamat penumpang Titanic.
    """
    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        # Buat DataFrame dari input
        input_data = pd.DataFrame([{
            'Pclass'    : passenger.Pclass,
            'Sex'       : passenger.Sex,
            'Age'       : passenger.Age,
            'SibSp'     : passenger.SibSp,
            'Parch'     : passenger.Parch,
            'Fare'      : passenger.Fare,
            'FamilySize': passenger.FamilySize,
            'IsAlone'   : passenger.IsAlone,
            # One-hot kolom tambahan (default 0 jika tidak diisi)
            'Embarked_Q': 0,
            'Embarked_S': 1,
            'AgeGroup_Senior' : 0,
            'AgeGroup_Teen'   : 0,
            'AgeGroup_Young Adult': 0,
            'AgeGroup_Adult'  : 0,
        }])

        # Align kolom dengan model (tambah kolom yang kurang, hapus yang lebih)
        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model.feature_names_in_]

        # Prediksi
        pred       = int(model.predict(input_data)[0])
        prob       = model.predict_proba(input_data)[0]
        prob_surv  = float(prob[1])
        prob_not   = float(prob[0])
        latency_ms = (time.time() - start) * 1000

        # ── Update Prometheus metrics ──────────
        REQUEST_COUNT.labels(
            method='POST', endpoint='/predict', status='200'
        ).inc()
        REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start)
        PREDICTION_PROBABILITY.observe(prob_surv)
        PREDICTION_CLASS_COUNT.labels(
            predicted_class='survived' if pred == 1 else 'not_survived'
        ).inc()

        return PredictionOutput(
            survived              = pred,
            survived_label        = 'Survived ✅' if pred == 1 else 'Not Survived ❌',
            probability_survive   = round(prob_surv, 4),
            probability_not_survive= round(prob_not, 4),
            latency_ms            = round(latency_ms, 2)
        )

    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()


# ──────────────────────────────────────────────
# ENDPOINT: /metrics (Prometheus scrape)
# ──────────────────────────────────────────────
@app.get('/metrics', tags=['Monitoring'])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ──────────────────────────────────────────────
# ENDPOINT: /health
# ──────────────────────────────────────────────
@app.get('/health', tags=['Health'])
async def health():
    """Health check endpoint."""
    return {
        'status' : 'healthy',
        'app'    : APP_NAME,
        'version': APP_VERSION,
        'author' : APP_AUTHOR,
        'model_loaded': model is not None
    }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5001, reload=False)
