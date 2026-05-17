# Membangun_model - Devin Novansyah

Folder ini berisi script pemodelan Machine Learning menggunakan **MLflow Tracking UI**.

## Struktur Folder

```
Membangun_model/
├── modelling.py                    ← Basic: autolog tanpa tuning
├── modelling_tuning.py             ← Skilled: manual log + GridSearchCV
├── titanic_preprocessing/
│   └── train_preprocessed.csv     ← Dataset hasil preprocessing (copy dari Kriteria 1)
├── screenshoot_dashboard.jpg       ← Screenshot MLflow UI (isi sendiri)
├── screenshoot_artifak.jpg         ← Screenshot artefak MLflow (isi sendiri)
└── requirements.txt
```

## Cara Menjalankan

### Install dependencies
```bash
pip install -r requirements.txt
```

### Basic — modelling.py (autolog)
```bash
python modelling.py
```

### Skilled — modelling_tuning.py (manual log + tuning)
```bash
python modelling_tuning.py
```

### Lihat hasil di MLflow UI
```bash
mlflow ui
# Buka: http://127.0.0.1:5000
```

## Apa yang di-log MLflow?

### modelling.py (autolog)
- Parameter model (n_estimators, max_depth, dll)
- Metrik: accuracy, precision, recall, f1
- Model artifact

### modelling_tuning.py (manual logging)
| Kategori | Detail |
|----------|--------|
| **Parameter** | Best params dari GridSearchCV, test_size, random_state, cv_folds |
| **Metrik** | accuracy, precision, recall, f1_score, roc_auc, cv_mean_accuracy, cv_std_accuracy, best_cv_score |
| **Artefak** | confusion_matrix.png, feature_importance.png, classification_report.txt, best_params.json |

## Author
**Devin Novansyah** - Machine Learning Submission
