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
import io
import zipfile

# --- CONFIGURACIÓN DE RUTAS ---
# Añadimos las carpetas de los modelos al path para poder importar sus módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
classifier_dir = os.path.join(current_dir, "..", "Tumor_Classification")
segmentation_dir = os.path.join(current_dir, "..", "Tumor_Segmentation")

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
    - Arquitectura: **CNN convolucional**
    - Tarea: **Clasificación de tipo de tumor**
    - Nº de clases: **4** (Glioma, Meningioma, Pituitary, No Tumor)
    - Accuracy en el conjunto de prueba: **{CLASSIFIER_ACCURACY*100:.1f}%**
    """
)

st.sidebar.subheader("Modelo de segmentación")
st.sidebar.markdown(
    f"""
    - Arquitectura: **U-Net para segmentación**
    - Tarea: **Segmentación de la región tumoral**
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
    st.image(image, caption="Imagen subida", width = 300)

    # Botón para iniciar el pipeline
    if st.button("🚀 Procesar imagen"):
        with st.spinner("Procesando imagen..."):

            # Circulito / etapa de inicio de procesamiento
            st.markdown("### 🔵 Iniciando del procesamiento de la imagen...")

            # PREPROCESAMIENTO
            st.markdown("#### 🧠 Paso 1: Preprocesamiento")
            from funciones import preprocess_for_classifier, preprocess_for_segmenter

            st.info(f"Imagen preprocesada para clasificación y segmentación.")

            # CARGA DE MODELOS
            classifier = load_classifier()
            segmenter = load_segmenter()

            mask_image = None
            mask_image_large = None

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
                from funciones import overlay_mask_on_image, keep_largest_component
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

                        # Mantener sólo la componente más grande
                        largest_mask = keep_largest_component(output_seg) 
                        mask_array = np.array(largest_mask * 255, dtype=np.uint8)
                        mask_image_large = Image.fromarray(mask_array, 'L')

                        # Imagen con máscara superpuesta en rojo
                        overlay_image = overlay_mask_on_image(image, mask_image_large, color=(255, 0, 0), alpha=0.4)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(mask_image, caption="Máscara de Segmentación", width=300)
                        with col2:
                            st.image(overlay_image, caption="Imagen con máscara superpuesta", width=300)
                        

                st.success("¡Procesamiento completado!")
                # Para descargar resultados
                st.markdown("#### 📥 Descargar Resultados")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    # 1) Imagen original
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format="PNG")
                    zip_file.writestr("imagen_original.png", img_bytes.getvalue())

                    # 2) Máscara de segmentación (solo si existe)
                    if mask_image is not None:
                        mask_bytes = io.BytesIO()
                        mask_image.save(mask_bytes, format="PNG")
                        zip_file.writestr("mascara_segmentacion.png", mask_bytes.getvalue())

                    # Imágen con máscara superpuesta (solo si existe)
                    if mask_image_large is not None:
                        overlay_image_bytes = io.BytesIO()
                        overlay_image.save(overlay_image_bytes, format="PNG")
                        zip_file.writestr("imagen_con_mascara_superpuesta.png", overlay_image_bytes.getvalue())

                    # 3) TXT con resultado y disclaimer
                    txt_lines = [
                        f"Resultado de la clasificación: {predicted_label}",
                        "",
                        "Información adicional:",
                        f"- Tumor detectado: {'Sí' if predicted_label != 'No Tumor' else 'No'}",
                        "",
                        "Aviso importante:",
                        "Este resultado fue generado por un modelo de inteligencia artificial.",
                        "No constituye un diagnóstico médico.",
                        "Para cualquier decisión o duda sobre tu salud, consulta siempre",
                        "a un/a profesional médico/a."
                    ]
                    zip_file.writestr("resultado_clasificacion.txt", "\n".join(txt_lines))

            zip_buffer.seek(0)

            st.download_button(
                label="⬇️ Descargar resultados (.zip)",
                data=zip_buffer.getvalue(),
                file_name="resultado_tumor.zip",
                mime="application/zip"
            )

