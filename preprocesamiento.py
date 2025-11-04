# Código para preprocesamiento de datos: se realiza redimensionamiento.
# Agregar funciones adicionales según sea necesario.

from PIL import Image
import numpy as np
from typing import Union
from pathlib import Path
import io

def redimensionar(imagen_entrada: Union[str, Path, io.BytesIO]) -> np.ndarray:
    """
    Carga una imagen 2D ( MRI en PNG/JPG), mantiene su modo (grises o color)
    y la redimensiona a 512x512.

    Parámetros
    ----------
    imagen_entrada : str | Path | io.BytesIO
        Ruta a la imagen o archivo en memoria (st.file_uploader en Streamlit).

    Retorna
    -------
    np.ndarray
        Imagen 2D (512, 512, C) si es color, o (512, 512) si es en grises.
    """
    # Cargar imagen (acepta ruta o buffer tipo archivo)
    img = Image.open(imagen_entrada)

    # Redimensionar a 512x512 (sin cambiar el modo)
    img = img.resize((512, 512), Image.BILINEAR)

    # Devolver como arreglo numpy
    return np.array(img)

