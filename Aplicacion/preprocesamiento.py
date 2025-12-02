# Código para preprocesamiento de datos: se realiza redimensionamiento.
# Agregar funciones adicionales según sea necesario.

from PIL import Image
import numpy as np
from typing import Union
from pathlib import Path
import io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def preprocess_for_classifier(image):
    """Preprocesa la imagen para el clasificador."""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def preprocess_for_segmenter(image):
    """Preprocesa la imagen para el segmentador."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((512, 512))
    ])
    img_tensor = TF.to_tensor(transform(image))
    return img_tensor.unsqueeze(0) # Batch dimension
