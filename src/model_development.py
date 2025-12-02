from src.data_preparation import DataPreparation
import xgboost as xgb
from src.config import settings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

import os
import numpy as np


class ModelDevelopment:
  def __init__(self):
    self.data = None
    self.data_train = None
    self.data_validation = None

    try:
      data_preparation = DataPreparation()
      self.data = data_preparation.get_output()

      if not os.path.exists(settings.MODEL_PATH):
        os.makedirs(settings.MODEL_PATH)

    except Exception as e:
      print(f"❌ Error al obtener los datos desde DataPreparation: {e}")

  def training_and_evaluation(self):
    """
    Realiza el entrenamiento y evaluación del modelo USANDO VALIDACIÓN CRUZADA,
    como se menciona en el paper.
    """
    X = self.data.drop(columns=[settings.FEATURE_TARGET])
    y = self.data[settings.FEATURE_TARGET]

    # 1. Inicializar el modelo con hiperparámetros del paper
    xgb_model = xgb.XGBClassifier(**settings.MODEL_HYPERPARAMETERS)

    # 2. Definir la estrategia de validación cruzada
    k_folds = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=settings.RANDOM_STATE)

    print(f"🚀 Iniciando validación cruzada (k={k_folds})...")

    # 3. Evaluar el modelo usando validación cruzada
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

    from sklearn.model_selection import cross_validate

    scores = cross_validate(xgb_model, X, y, cv=skf, scoring=scoring_metrics)

    print("✅ Validación cruzada completada.")
    print(f"\n📊 Métricas Promedio (k={k_folds} folds):")
    print(f"  - Accuracy:  {np.mean(scores['test_accuracy']):.4f}")
    print(f"  - Precision: {np.mean(scores['test_precision']):.4f}")
    print(f"  - Recall:    {np.mean(scores['test_recall']):.4f}")
    print(f"  - F1-Score:  {np.mean(scores['test_f1']):.4f}")

    print("\n🚀 Entrenando modelo final en el 100% de los datos...")
    final_model = xgb.XGBClassifier(**settings.MODEL_HYPERPARAMETERS)
    final_model.fit(X, y)

    self.model = final_model
    model_save_path = os.path.join(settings.MODEL_PATH, "xgb_model.json")
    self.model.save_model(model_save_path)
    print(f"💾 Modelo final guardado en: {model_save_path}")

    return scores
