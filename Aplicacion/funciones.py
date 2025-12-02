# Código para preprocesamiento de datos: se realiza redimensionamiento.
# Agregar funciones adicionales según sea necesario.

from PIL import Image
import numpy as np
from typing import Union
from pathlib import Path
import io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import io
import zipfile
from scipy import ndimage

## Funciones de preprocesamiento

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

## Funciones auxiliares

def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Recibe una máscara binaria 2D (bool o 0/1) y devuelve sólo
    la componente conexa más grande. Si no hay componentes, devuelve la original.
    """
    # Etiquetado de componentes conectadas
    labeled, num_components = ndimage.label(binary_mask)

    if num_components == 0:
        # No hay nada segmentado -> devolvemos tal cual
        return binary_mask

    # Tamaño (número de píxeles) de cada componente
    component_sizes = ndimage.sum(binary_mask, labeled, index=range(1, num_components + 1))

    # Índice de la componente más grande (los labels empiezan en 1)
    largest_label = int(np.argmax(component_sizes)) + 1

    # Nueva máscara: True sólo donde está la componente más grande
    largest_component = (labeled == largest_label)

    return largest_component


def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.4):
    """
    Superpone la máscara sobre la imagen original.
    - image: PIL.Image RGB (imagen original)
    - mask:  PIL.Image en escala de grises (0–255) o array equivalente
    - color: color de la máscara (R, G, B)
    - alpha: transparencia de la máscara
    """
    # Asegurar RGB
    image = image.convert("RGB")

    # Redimensionar la imagen al tamaño de la máscara (512x512)
    image = image.resize(mask.size)

    img_np = np.array(image).astype(np.float32)
    mask_np = np.array(mask)

    # Máscara binaria
    mask_bin = mask_np > 0  # True donde hay tumor

    # Copia de la imagen para el overlay
    overlay = img_np.copy()

    # Aplicamos el color sólo donde hay máscara
    overlay[mask_bin] = (
        (1 - alpha) * overlay[mask_bin] + alpha * np.array(color, dtype=np.float32)
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)
