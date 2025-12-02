# Predicción de Riesgo de Hipertensión

Un pipeline modular de aprendizaje automático diseñado para predecir el riesgo de hipertensión utilizando datos clínicos. Este proyecto implementa la ingesta de datos, el preprocesamiento, el entrenamiento de modelos con XGBoost y el análisis de explicabilidad mediante valores SHAP.

## Dataset

https://zenodo.org/records/4567767

## Estructura del Proyecto

* `src/`: Contiene los módulos de código fuente.
    * `data_collection.py`: Maneja la carga de datos crudos.
    * `data_preparation.py`: Realiza la limpieza y transformación de datos.
    * `model_development.py`: Gestiona el entrenamiento del modelo, la validación cruzada y el guardado.
    * `config.py`: Almacena variables de configuración y rutas de archivos.
* `models/`: Directorio para los artefactos del modelo guardado.
* `data/`: Directorio para conjuntos de datos crudos y procesados.
* `main.py`: Punto de entrada para ejecutar el pipeline de entrenamiento.
* `evaluation.ipynb`: Cuadernos Jupyter para el análisis SHAP.

## Requisitos Previos

* Python 3.11
* Conda (recomendado)

## Instalación

1.  Clonar el repositorio en su máquina local.
2.  Crear el entorno de Conda utilizando el archivo de configuración proporcionado. Esto asegura la instalación de versiones compatibles de `numba` y `llvmlite`.

```bash
conda env create -f environment.yml
```

1.  Activar el entorno.

```bash
conda activate hypertension-prediction
```

## Uso

### Entrenamiento del Modelo

Para ejecutar el pipeline completo (procesamiento de datos, entrenamiento y evaluación), ejecute el script principal:

```bash
python main.py
```

Este proceso realizará lo siguiente:

1.  Cargar y limpiar el conjunto de datos.
2.  Ejecutar una Validación Cruzada Estratificada (Stratified K-Fold, k=10).
3.  Entrenar el modelo final con el 100% de los datos.
4.  Guardar el modelo entrenado en `models/xgb_model.json`.

### Análisis y Explicabilidad

Para generar gráficos SHAP y analizar la importancia de las variables:

1.  Abrir `evaluation.ipynb`.
2.  Asegurarse de que el kernel esté configurado como `hypertension-prediction`.
3.  Ejecutar la celdas para generar los gráficos de resumen (summary plots), gráficos de dependencia y gráficos de fuerza (force plots).

## Métricas Clave

* **Modelo:** Clasificador XGBoost
* **Validación:** K-Fold Estratificado (k=10)
* **Explicabilidad:** SHAP (SHapley Additive exPlanations) TreeExplainer