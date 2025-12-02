from src.data_collection import DataCollection
from src.config import settings


class DataPreparation:
  def __init__(self):
    self.data = None

    try:
      data_collection = DataCollection()
      self.data = data_collection.get_output()
    except Exception as e:
      print(f"❌ Error al obener los datos desde DataCollection: {e}")

  def data_cleaning(self):
    """
    Realiza la limpieza de los datos.
    """
    print(f"🧹 Filas antes de limpiar: {len(self.data)}")
    self.data = self.data.dropna()
    self.data = self.data.query('hypertension_dx != "."')
    self.data = self.data.reset_index(drop=True)

    print(f"✅ Filas después de limpiar: {len(self.data)}")

    return self.data

  def data_transformation(self):
    """
    Realiza la transformación de los datos.
    """
    for feature in settings.FEATURES_TRANSFORMATION_MAP:
      self.data[feature] = self.data[feature].map(settings.FEATURES_TRANSFORMATION_MAP[feature])

    return self.data

  def get_output(self):
    """
    Retorna la salida de esta etapa.
    """
    self.data_cleaning()
    self.data_transformation()
    return self.data
