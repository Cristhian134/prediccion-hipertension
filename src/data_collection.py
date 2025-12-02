import os
import requests

import pandas as pd

from src.config import settings


class DataCollection:
  """
  Clase para la recolección y adquisición de datos.
  """

  def raw_data_acquisition(self):
    """
    Descarga los datos crudos desde una URL y los guarda localmente.
    """

    output_filepath = os.path.join(settings.DATA_PATH, settings.RAW_DATA_NAME)

    # Crea la carpeta 'data/' si no existe
    os.makedirs(settings.DATA_PATH, exist_ok=True)

    print(f"⬇️ Iniciando descarga desde: {settings.RAW_DATA_URL}")

    try:
      response = requests.get(settings.RAW_DATA_URL, timeout=60)
      response.raise_for_status()

      with open(output_filepath, 'wb') as file:
        file.write(response.content)

      print(f"✅ Descarga completada. Archivo guardado en: {output_filepath}")

    except Exception as e:
      print(f"❌ Error al descargar el archivo: {e}")

  def data_discovery_and_selection(self):
    """
    Descubrimiento y selección de datos inicial, revisar el notebook.
    """
    df = pd.read_excel(os.path.join(settings.DATA_PATH, settings.RAW_DATA_NAME), sheet_name=0)
    X = df.drop(columns=settings.FEATURES_DROP_INITIAL)

    return X

  def get_output(self):
    """
    Retorna la salida de esta etapa.
    """
    return self.data_discovery_and_selection()
