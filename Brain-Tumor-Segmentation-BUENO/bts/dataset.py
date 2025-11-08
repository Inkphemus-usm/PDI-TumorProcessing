from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random

class TumorDataset(Dataset):
    """ 
    Versión modificada para leer el dataset con formato 'enh_X.png' 
    y 'enh_X_mask.png', manejando índices no secuenciales.
    """

    def __init__(self, root_dir, transform=True, DEBUG=False):
        """ Constructor modificado """
        self.root_dir = root_dir
        self.transform = {'hflip': TF.hflip,
                          'vflip': TF.vflip,
                          'rotate': TF.rotate}
        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])
        self.DEBUG = DEBUG
        if not transform:
            self.transform = None

        # --- LÓGICA MODIFICADA ---
        # 1. Escanear el directorio y encontrar todos los pares válidos
        self.image_files = []
        all_files = set(os.listdir(self.root_dir)) # Usar un set para búsquedas rápidas

        # 2. Encontrar solo los archivos de IMAGEN (sin máscara)
        image_filenames = [f for f in all_files if f.startswith('enh_') and f.endswith('.png') and not f.endswith('_mask.png')]

        # 3. Verificar que cada imagen tenga su máscara correspondiente
        for img_name in image_filenames:
            mask_name = img_name.replace('.png', '_mask.png')
            if mask_name in all_files:
                # Solo añadir la imagen si su par de máscara existe
                self.image_files.append(img_name)
        
        # 4. Ordenar la lista de archivos numéricamente, no alfabéticamente
        # (ej: 'enh_10.png' debe ir después de 'enh_2.png')
        self.image_files = sorted(self.image_files, 
                                  key=lambda f: int(f.replace('enh_', '').replace('.png', '')))
        
        if self.DEBUG:
            print(f"Se encontraron {len(self.image_files)} pares de imagen/máscara.")
        # --- FIN DE LA MODIFICACIÓN ---

    def __getitem__(self, index):
        """ 
        Función modificada para cargar por índice de la lista 
        en lugar de construir el nombre.
        """
        
        # --- LÓGICA MODIFICADA ---
        # Obtener el nombre base del archivo de nuestra lista ordenada
        image_name_base = self.image_files[index] # ej: 'enh_5.png'
        
        # Construir el nombre de la máscara correspondiente
        mask_name_base = image_name_base.replace('.png', '_mask.png') # ej: 'enh_5_mask.png'

        # Construir las rutas completas
        image_path = os.path.join(self.root_dir, image_name_base)
        mask_path = os.path.join(self.root_dir, mask_name_base)
        
        # Extraer el índice numérico del nombre (para el dict de 'sample')
        # ej: 'enh_5.png' -> 5
        try:
            file_index = int(image_name_base.replace('enh_', '').replace('.png', ''))
        except ValueError:
            print(f"Error al procesar el nombre: {image_name_base}")
            file_index = index # Usar índice de lista como respaldo
        # --- FIN DE LA MODIFICACIÓN ---

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.default_transformation(image)
        mask = self.default_transformation(mask)

        # Custom transformations
        if self.transform:
            image, mask = self._random_transform(image, mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        mask = (mask > 0.5).float()

        sample = {'index': file_index, 'image': image, 'mask': mask}
        return sample

    def _random_transform(self, image, mask):
        """ 
        Aplica un set de transformaciones en orden aleatorio.
        (Esta función no necesita cambios)
        """
        choice_list = list(self.transform)
        for _ in range(len(choice_list)):
            choice_key = random.choice(choice_list)
            if self.DEBUG:
                print(f'Transform choose: {choice_key}')
            action_prob = random.randint(0, 1)
            if action_prob >= 0.5:
                if self.DEBUG:
                    print(f'\tApplying transformation: {choice_key}')
                if choice_key == 'rotate':
                    rotation = random.randint(15, 75)
                    if self.DEBUG:
                        print(f'\t\tRotation by: {rotation}')
                    image = self.transform[choice_key](image, rotation)
                    mask = self.transform[choice_key](mask, rotation)
                else:
                    image = self.transform[choice_key](image)
                    mask = self.transform[choice_key](mask)
            choice_list.remove(choice_key)

        return image, mask

    def __len__(self):
        """ 
        Función modificada para devolver el número 
        de pares de imagen/máscara que se encontraron.
        """
        # --- LÓGICA MODIFICADA ---
        return len(self.image_files)
        # --- FIN DE LA MODIFICACIÓN ---