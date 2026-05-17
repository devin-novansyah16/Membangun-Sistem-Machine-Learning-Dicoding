"""
modelling.py
=============
Melatih model machine learning menggunakan MLflow Tracking UI
dengan autolog (tanpa hyperparameter tuning).

Author  : Devin Novansyah
Dataset : Titanic - Machine Learning from Disaster
Model   : Random Forest Classifier
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(__file__),
                            'titanic_preprocessing', 'train_preprocessed.csv')
EXPERIMENT   = 'Titanic_Modelling_Devin-Novansyah'
RUN_NAME     = 'RandomForest_Autolog'
TARGET_COL   = 'Survived'
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_data(path: str):
    print(f'[1/4] Memuat data dari: {path}')
    df = pd.read_csv(path)

    # Pastikan hanya kolom numerik
    df = df.select_dtypes(include='number')

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f'      ✅ Shape X: {X.shape} | Shape y: {y.shape}')
    return X, y


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train(X, y):
    print('[2/4] Membagi data (train/test split)...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f'      Train: {X_train.shape} | Test: {X_test.shape}')

    # Setup MLflow
    mlflow.set_experiment(EXPERIMENT)

    print('[3/4] Memulai MLflow run dengan autolog...')
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print(f'      Run ID: {run.info.run_id}')

        # Inisialisasi & latih model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        # Evaluasi manual (ditampilkan di console)
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        print('[4/4] Hasil Evaluasi:')
        print(f'      Accuracy  : {acc:.4f}')
        print(f'      Precision : {prec:.4f}')
        print(f'      Recall    : {rec:.4f}')
        print(f'      F1-Score  : {f1:.4f}')
        print(f'\n✅ Run selesai! Cek MLflow UI dengan: mlflow ui')

    return model


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print('  MODELLING - TITANIC (Autolog)')
    print('  Author: Devin Novansyah')
    print('=' * 55 + '\n')

    X, y = load_data(DATA_PATH)
    train(X, y)
