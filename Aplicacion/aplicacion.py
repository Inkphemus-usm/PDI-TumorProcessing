# Configuración de Streamlit para la aplicación web

# Librerias
import os
import sys
import streamlit as st
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# --- CONFIGURACIÓN DE RUTAS ---
# Añadimos las carpetas de los modelos al path para poder importar sus módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
classifier_dir = os.path.join(current_dir, "..\Tumor_Classification")
segmentation_dir = os.path.join(current_dir, "..\Tumor_Segmentation")

if classifier_dir not in sys.path:
    sys.path.append(classifier_dir)
if segmentation_dir not in sys.path:
    sys.path.append(segmentation_dir)

# Importar los modelos
# Nota: src.model es del clasificador, bts.model es del segmentador
try:
    from src.model import MyModel
    from src.utils import predict as predict_classifier
    from bts.model import DynamicUNet
except ImportError as e:
    st.error(f"Error al importar módulos: {e}. Verifique que las carpetas de los modelos existan y tengan los archivos necesarios.")
    st.stop()

# --- CONFIGURACIÓN DE DISPOSITIVOS ---
device_classifier = "cuda" if torch.cuda.is_available() else "cpu"
device_segmentation = "cpu" # Forzado a CPU según recomendación del código original

# --- CARGA DE MODELOS (CACHED) ---

@st.cache_resource
def load_classifier():
    """Carga el modelo de clasificación."""
    model_path = os.path.join(classifier_dir, "models", "model_30")
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo de clasificación en: {model_path}")
        return None
    
    model = MyModel(num_classes=5)
    # map_location asegura que cargue en el dispositivo correcto
    model.load_state_dict(torch.load(model_path, map_location=device_classifier))
    model.to(device_classifier)
    model.eval()
    return model

@st.cache_resource
def load_segmenter():
    """Carga el modelo de segmentación."""
    model_path = os.path.join(segmentation_dir, "saved_models", "UNet-Reentrenado-v6.pt")
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo de segmentación en: {model_path}")
        return None

    filter_list = [8, 16, 32, 64, 128]
    model = DynamicUNet(filter_list).to(device_segmentation)
    
    state_dict = torch.load(model_path, map_location=device_segmentation)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- DICCIONARIO DE ETIQUETAS ---
label_dict = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary",
    3: "No Tumor"
}

# --- MÉTRICAS DE LOS MODELOS (puedes ajustarlas) ---
CLASSIFIER_ACCURACY = 0.9817 
SEGMENTER_ACCURACY  = 0.732

# --- INTERFAZ DE USUARIO ---

st.title("Bienvenido/a a la Clasificación y Segmentación de tumores cerebrales 🧠")

st.sidebar.title("📊 Información de los modelos")

st.sidebar.subheader("Modelo de clasificación")
st.sidebar.markdown(
    f"""
    - Arquitectura: `MyModel` personalizado
    - Nº de clases: **4** (Glioma, Meningioma, Pituitary, No Tumor)
    - Accuracy en el conjunto de prueba: **{CLASSIFIER_ACCURACY*100:.1f}%**
    """
)

st.sidebar.subheader("Modelo de segmentación")
st.sidebar.markdown(
    f"""
    - Arquitectura: `DynamicUNet`
    - Accuracy en el conjunto de prueba: **{SEGMENTER_ACCURACY*100:.1f}%**
    """
)

st.sidebar.markdown("---")

st.sidebar.warning(
    "⚠️ **Aviso importante**\n\n"
    "Los resultados mostrados por esta aplicación son generados por modelos "
    "de inteligencia artificial y **no son 100% precisos**. "
    "Esta herramienta tiene fines académicos/demostrativos y **no debe utilizarse "
    "para diagnóstico médico**.\n\n"
    "Para cualquier duda o confirmación sobre tu salud, consulta siempre con "
    "un/a profesional médico/a."
)


with st.expander("📖 Instrucciones de uso (click para desplegar)"):
    st.markdown(
        """
        1. Sube una imagen en formato JPG/PNG utilizando el botón de carga.
        2. La aplicación procesará la imagen (esto puede tomar varios segundos).
        3. Se mostrará la predicción de la clase del tumor cerebral o la ausencia de uno.
        4. Si se detecta un tumor, se generará y mostrará la máscara de segmentación.
        5. Puedes repetir el proceso con diferentes imágenes según sea necesario.
        6. (Opcional) Descarga los resultados si la opción está disponible.
        """
    )


# image from user
uploaded_file = st.file_uploader("Seleccione su imagen médica...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("¡Imagen cargada con éxito!")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", width = 210)

    # Botón para iniciar el pipeline
    if st.button("🚀 Procesar imagen"):
        with st.spinner("Procesando imagen..."):

            # Circulito / etapa de inicio de procesamiento
            st.markdown("### 🔵 Iniciando del procesamiento de la imagen...")

            # PREPROCESAMIENTO
            st.markdown("#### 🧠 Paso 1: Preprocesamiento")
            from preprocesamiento import preprocess_for_classifier, preprocess_for_segmenter

            st.info(f"Imagen preprocesada para clasificación y segmentación.")

            # CARGA DE MODELOS
            classifier = load_classifier()
            segmenter = load_segmenter()

            if classifier is None or segmenter is None:
                st.error("No se pudieron cargar los modelos. Revise los logs.")
            else:
                # CLASIFICACIÓN
                st.markdown("#### 🧠 Paso 2: Clasificación")
                
                input_classifier = preprocess_for_classifier(image).to(device_classifier)
                
                # Inferencia
                with torch.no_grad():
                    outputs = classifier(input_classifier)
                    _, predicted_idx = torch.max(outputs, 1)
                    predicted_label = label_dict[predicted_idx.item()]
                
                st.info(f"Resultado de la clasificación: **{predicted_label}**")

                # SEGMENTACIÓN
                st.markdown("#### 🧠 Paso 3: Segmentación")
                
                if predicted_label == "No Tumor":
                    st.warning("No se detectó tumor, se omite la segmentación.")
                else:
                    st.write(f"Detectado {predicted_label}, procediendo a segmentar...")
                    
                    input_segmenter = preprocess_for_segmenter(image).to(device_segmentation)
                    
                    with torch.no_grad():
                        # El modelo espera [batch, channel, height, width]
                        # input_segmenter ya tiene batch dim (1, 1, 512, 512)
                        output_seg = segmenter(input_segmenter).detach().cpu()
                        output_seg = (output_seg > 0.5) # Binarización
                        output_seg = output_seg.numpy()[0, 0] # Remove batch and channel dims
                        
                        # Post-procesamiento para visualización
                        mask_image = np.array(output_seg * 255, dtype=np.uint8)
                        mask_image = Image.fromarray(mask_image, 'L')
                        
                        st.image(mask_image, caption="Máscara de Segmentación", width=210)

                st.success("¡Procesamiento completado!")
                # Opción para descargar resultados (Pendiente de implementar lógica de guardado si se requiere)

