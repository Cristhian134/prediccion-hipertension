class Settings:
  # Configuración para la recolección de datos
  DATA_PATH = "data"
  RAW_DATA_NAME = "hypertension_dataset.xls"
  RAW_DATA_URL = "https://zenodo.org/records/4567767/files/Change%20criteria%20hypertension%20peru.xls?download=1"

  FEATURES_DROP_INITIAL = [
    "id",
    "city",
    "masl",
    # "systolic_bp",
    # "diastolic_bp",
    "smoking_years",
    "hypertension_years",
    "hypertension_treatment",
    "msnm",
    "region",
    "sist_old",
    "diast_old",
    "sist_new",
    "diast_new",
    "treatment ",
    "HTA_new",
    "cv_diseases",
    "dm_treatment",
    "BMI_cat",
    "cd_treatment",
    "height_cm",
    "weight_kg",
  ]
  FEATURE_TARGET = "hypertension_dx"

  FEATURES_TRANSFORMATION_MAP = {
    "sex": {
      "Female": 0,
      "Male": 1,
    },
    "diabetes_mellitus": {
      "No": 0,
      "Yes": 1,
    },
    "smoking": {
      "No": 0,
      "Yes": 1,
    },
    "physical_activity": {
      "No": 0,
      "Yes": 1,
    },
    "hypertension_dx": {
      "No": 0,
      "Yes": 1,
    },
  }

  # Configuración para los modelos
  MODEL_PATH = "models"

  MODEL_HYPERPARAMETERS = {
    "eta": 0.3,                 # Tasa de aprendizaje
    "n_estimators": 100,        # Número de árboles
    "gamma": 0,                 # Mínima reducción de pérdida para hacer una partición
    "max_depth": 6,             # Profundidad máxima de un árbol
    "min_child_weight": 1,      # Mínima suma de pesos de instancia necesaria en un hijo
    "max_delta_step": 0,
    "subsample": 1,             # Proporción de la muestra de entrenamiento
    "sampling_method": "uniform",
  }

  RANDOM_STATE = 42

  TEST_SET_SIZE = 0.3
  TRAIN_SET_SIZE = 0.7


settings = Settings()
