import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

# Asegúrate de que estos archivos estén correctamente ubicados
from bts.model import DynamicUNet 
from bts.classifier import BrainTumorClassifier 


def get_arguments():
    """Returns the command line arguments as a dict"""
    parser = argparse.ArgumentParser()
    # Usamos '--folder' en lugar de '--dir' para consistencia interna
    parser.add_argument('--file', required=False, type=str,
                        help='Single input file name.')
    parser.add_argument('--folder', required=False,
                        type=str, help='Directory name with input images')
    parser.add_argument('--ofp', required=False,
                        type=str, help='Single output file path with name. Use this if using "file" flag.')
    parser.add_argument('--odp', required=False,
                        type=str, help='Directory path for output images. Use this if using "folder" flag.')
    args = parser.parse_args()
    args = {'file': args.file, 'folder': args.folder, 
            'ofp': args.ofp, 'odp': args.odp}
    return args


class Api:
    def __init__(self):
        # CAMBIO 1: FORZAMOS EL USO DE CPU para evitar el error cuDNN/GPU
        self.device = torch.device('cpu') 
        print(f"Dispositivo de ejecución forzado a: {self.device}")

    def call(self, file, folder, ofp, odp):
        """Method saves the predicted image by taking different parameters."""
        if file != None and folder != None:
            print('"folder" flag and "file" flag cant be used together')
            return

        model = self._load_model()
        save_path = None
        
        # Para un solo archivo
        if file != None:
            # Verificación de existencia del archivo
            if not os.path.exists(file):
                 print(f"ERROR: Archivo no encontrado en la ruta: {file}")
                 return

            image = self._get_file(file)
            output = self._get_model_output(image, model)

            # --- Manejo del nombre de archivo de salida ---
            name_base = os.path.basename(file)
            name_only, extension = os.path.splitext(name_base)
            save_name = name_only + '_predicted' + '.png' # Usamos .png por ser mejor para máscaras

            save_path = save_name 
            if ofp:
                # Si ofp es una carpeta, se une con el nombre
                if os.path.isdir(ofp) or not os.path.splitext(ofp)[1]: 
                    save_path = os.path.join(ofp, save_name)
                # Si ofp es un nombre de archivo completo, se usa directamente
                else:
                    save_path = ofp
                
            self._save_image(output, save_path)
            print(f'Output Image Saved At {save_path}')

        elif folder != None:
            # Verificación de existencia de la carpeta
            if not os.path.isdir(folder):
                print(f"ERROR: Carpeta de entrada no encontrada en la ruta: {folder}")
                return
            
            image_list = os.listdir(folder)
            
            # Asegurar que el directorio de salida exista
            if odp:
                os.makedirs(odp, exist_ok=True)
            
            for file_name in image_list:
                file = os.path.join(folder, file_name)
                
                # Omitir archivos que no sean de imagen (evita errores con archivos de sistema)
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                image = self._get_file(file)
                output = self._get_model_output(image, model)

                # Generación del nombre de archivo de salida
                name, extension = os.path.splitext(file_name)
                save_name = name+'_predicted'+'.png'

                # Definición de la ruta de guardado
                if odp:
                    save_path = os.path.join(odp, save_name)
                else:
                    save_path = os.path.join(folder, save_name)

                self._save_image(output, save_path)
                print(f'Output Image Saved At {save_path}')

    def _load_model(self):
        """Load the saved model and return it."""
        filter_list = [16, 32, 64, 128, 256]

        model = DynamicUNet(filter_list).to(self.device)
        # La clase BrainTumorClassifier es solo para envolver el modelo, no se usa para la carga de pesos
        classifier = BrainTumorClassifier(model, self.device) 
        model_path = os.path.join(
            'saved_models', 'UNet-[16, 32, 64, 128, 256].pt')
        
        # CAMBIO 2 y 3: Carga de pesos directamente con map_location=self.device
        # Esto soluciona el TypeError y el problema de cuDNN
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval() # Pone el modelo en modo de evaluación (sin entrenamiento)
        
        print(f'Saved model at location "{model_path}" loaded on {self.device}')
        return model

    def _get_model_output(self, image, model):
        """Returns the saved model output"""
        # Asegúrate de que la imagen está en el dispositivo correcto (CPU)
        with torch.no_grad(): # Desactiva el cálculo de gradientes para inferencia
            image = image.view((-1, 1, 512, 512)).to(self.device) 
            output = model(image).detach().cpu()
            output = (output > 0.5) # Binarización de la máscara de segmentación
            output = output.numpy()
            output = np.resize((output * 255), (512, 512)) # Escala a 0-255 y redimensiona
            return output

    def _save_image(self, image, path):
        """Save the image to storage specified by path"""
        # Se asegura de crear los directorios si no existen
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image = Image.fromarray(np.uint8(image), 'L') # Crea la imagen en escala de grises
        image.save(path)

    def _get_file(self, file_name):
        """Load the image by taking file name as input"""
        # Esta lógica es compatible con JPG y PNG (usa PIL)
        default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)) # Redimensiona al tamaño esperado por el modelo
        ])

        # Image.open() funciona con archivos JPG, PNG y otros formatos comunes
        image = default_transformation(Image.open(file_name)) 
        return TF.to_tensor(image)


if __name__ == "__main__":
    args = get_arguments()
    api = Api()
    api.call(**args)