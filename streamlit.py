# Configuración de Streamlit para la aplicación web

# Librerias
import os
import streamlit as st
from PIL import Image

st.title("Bienvenido/a a la Clasificación y Segmentación de tumores cerebrales")

st.write("Instrucciones de uso:")
st.write("1. Sube una imagen en formato JPG utilizando el botón de carga.")
st.write("2. La aplicación procesará la imagen, puede tomar varios segundos.")
st.write("3. Se mostrará la predicción de la clase del tumor cerebral o la ausencia de uno.")
st.write("4. Se segmentará el tumor en la imagen si se detecta uno.")
st.write("5. Repite el proceso con diferentes imágenes según sea necesario.")
st.write("6. Descarga los resultados.")


# image from user
uploaded_file = st.file_uploader("Seleccione su imagen médica...", type="jpg")

if uploaded_file is not None:
    st.success("¡Imagen cargada con éxito!")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", width = 210)

    # Preprocesamiento de la imagen

    # Modelo de clasificación

    # Mostrar predicción de la clase

    # Modelo de segmentación

    # Mostrar imagen segmentada

    st.success("¡Procesamiento completado!")

    # Opción para descargar resultados