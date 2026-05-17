"""
modelling_tuning.py
====================
Melatih model machine learning dengan hyperparameter tuning
menggunakan GridSearchCV dan manual logging MLflow
(bukan autolog).

Author  : Devin Novansyah
Dataset : Titanic - Machine Learning from Disaster
Model   : Random Forest Classifier + GridSearchCV
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
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
RUN_NAME     = 'RandomForest_ManualLog_Tuning'
TARGET_COL   = 'Survived'
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Grid hyperparameter yang akan dicoba
PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth'   : [4, 6, 8],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}


# ──────────────────────────────────────────────
# HELPER: BUAT ARTEFAK
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, tmp_dir: str) -> str:
    """Buat dan simpan confusion matrix sebagai gambar."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Not Survived', 'Survived'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix - Titanic\nDevin Novansyah', fontsize=13)
    plt.tight_layout()
    path = os.path.join(tmp_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def plot_feature_importance(model, feature_names, tmp_dir: str) -> str:
    """Buat dan simpan feature importance plot."""
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(12, len(feature_names))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(top_n),
           importances[indices[:top_n]],
           color='steelblue', edgecolor='white')
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices[:top_n]],
                       rotation=40, ha='right', fontsize=10)
    ax.set_title('Feature Importance (Top 12)\nDevin Novansyah', fontsize=13)
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    path = os.path.join(tmp_dir, 'feature_importance.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def save_classification_report(y_true, y_pred, tmp_dir: str) -> str:
    """Simpan classification report ke file teks."""
    report = classification_report(y_true, y_pred,
                                   target_names=['Not Survived', 'Survived'])
    path = os.path.join(tmp_dir, 'classification_report.txt')
    with open(path, 'w') as f:
        f.write('Classification Report - Titanic\n')
        f.write('Author: Devin Novansyah\n')
        f.write('=' * 45 + '\n')
        f.write(report)
    return path


def save_best_params(params: dict, tmp_dir: str) -> str:
    """Simpan best hyperparameters ke JSON."""
    path = os.path.join(tmp_dir, 'best_params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)
    return path


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

def load_data(path: str):
    print(f'[1/5] Memuat data dari: {path}')
    df = pd.read_csv(path)
    df = df.select_dtypes(include='number')
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f'      ✅ Shape X: {X.shape} | Shape y: {y.shape}')
    return X, y


# ──────────────────────────────────────────────
# HYPERPARAMETER TUNING + MANUAL LOGGING
# ──────────────────────────────────────────────

def train_with_tuning(X, y):
    print('[2/5] Membagi data (train/test split)...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f'      Train: {X_train.shape} | Test: {X_test.shape}')

    print('[3/5] Menjalankan GridSearchCV (hyperparameter tuning)...')
    base_model = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f'      ✅ Best params: {grid_search.best_params_}')
    print(f'      ✅ Best CV accuracy: {grid_search.best_score_:.4f}')

    # Prediksi & metrik
    y_pred      = grid_search.predict(X_test)
    y_pred_prob = grid_search.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    roc   = roc_auc_score(y_test, y_pred_prob)
    cv_sc = cross_val_score(grid_search.best_estimator_,
                            X_train, y_train, cv=5, scoring='accuracy')

    print('[4/5] Hasil Evaluasi:')
    print(f'      Accuracy       : {acc:.4f}')
    print(f'      Precision      : {prec:.4f}')
    print(f'      Recall         : {rec:.4f}')
    print(f'      F1-Score       : {f1:.4f}')
    print(f'      ROC-AUC        : {roc:.4f}')
    print(f'      CV Mean Acc    : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}')

    # ── MLflow Manual Logging ──────────────────
    print('[5/5] Logging ke MLflow (manual)...')
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print(f'      Run ID: {run.info.run_id}')

        # Tags
        mlflow.set_tags({
            'author'   : 'Devin Novansyah',
            'model'    : 'RandomForestClassifier',
            'tuning'   : 'GridSearchCV',
            'dataset'  : 'Titanic'
        })

        # Log hyperparameter terbaik
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param('test_size'   , TEST_SIZE)
        mlflow.log_param('random_state', RANDOM_STATE)
        mlflow.log_param('cv_folds'    , 5)

        # Log metrik
        mlflow.log_metric('accuracy'        , acc)
        mlflow.log_metric('precision'       , prec)
        mlflow.log_metric('recall'          , rec)
        mlflow.log_metric('f1_score'        , f1)
        mlflow.log_metric('roc_auc'         , roc)
        mlflow.log_metric('cv_mean_accuracy', cv_sc.mean())
        mlflow.log_metric('cv_std_accuracy' , cv_sc.std())
        mlflow.log_metric('best_cv_score'   , grid_search.best_score_)

        # Log model
        mlflow.sklearn.log_model(
            sk_model        = grid_search.best_estimator_,
            artifact_path   = 'model',
            input_example   = X_train.head(5)
        )

        # Log artefak tambahan
        with tempfile.TemporaryDirectory() as tmp:
            # Artefak 1: Confusion Matrix
            cm_path = plot_confusion_matrix(y_test, y_pred, tmp)
            mlflow.log_artifact(cm_path, artifact_path='plots')

            # Artefak 2: Feature Importance
            fi_path = plot_feature_importance(grid_search, X.columns.tolist(), tmp)
            mlflow.log_artifact(fi_path, artifact_path='plots')

            # Artefak 3: Classification Report
            cr_path = save_classification_report(y_test, y_pred, tmp)
            mlflow.log_artifact(cr_path, artifact_path='reports')

            # Artefak 4: Best Params JSON
            bp_path = save_best_params(grid_search.best_params_, tmp)
            mlflow.log_artifact(bp_path, artifact_path='reports')

        print('      ✅ Semua metrik, parameter, dan artefak berhasil di-log!')
        print(f'\n✅ Run selesai! Jalankan: mlflow ui')
        print(f'   Lalu buka: http://127.0.0.1:5000')

    return grid_search


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 55)
    print('  MODELLING TUNING - TITANIC (Manual Log)')
    print('  Author: Devin Novansyah')
    print('=' * 55 + '\n')

    X, y = load_data(DATA_PATH)
    train_with_tuning(X, y)
