# 🧠 PDI-TumorProcessing

**Plataforma web para la Clasificación y Segmentación de Tumores Cerebrales**, desarrollada en el contexto del curso de Procesamiento Digital de Imágenes.

La aplicación permite:

- Clasificar imágenes de resonancia magnética (MRI) según tipo de tumor.
- Segmentar la región tumoral y superponerla sobre la imagen original.
- Descargar los resultados (imagen original, máscara y clasificación) en un archivo comprimido.

> ⚠️ **Aviso importante**  
> Este proyecto tiene fines exclusivamente académicos y demostrativos.  
> Los resultados generados **no constituyen un diagnóstico médico** y no deben utilizarse para la toma de decisiones clínicas.  
> Para cualquier duda o evaluación de salud, es imprescindible consultar a un/a profesional médico/a.

---

## 👥 Integrantes

- Fabian Clavijo  
- Josefa Gómez  
- Diego Herrera  
- Julia Houdin  
- Alan Montero  

---

## 🧩 Modelos incluidos en el repositorio

1. **Tumor_Classification**  
   Modelo de *deep learning* para **clasificación de tumores cerebrales** a partir de imágenes MRI.  
   Clases consideradas:
   - Glioma  
   - Meningioma  
   - Pituitary (tumor hipofisario)  
   - No Tumor  

2. **Tumor_Segmentation**  
   Modelo de segmentación basado en una arquitectura tipo **U-Net**, entrenado para **delimitar la región tumoral** en imágenes MRI (máscara binaria).

3. **Aplicación (Streamlit)**  
   Interfaz web desarrollada con **Streamlit** que integra ambos modelos y permite:
   - Cargar una imagen MRI
   - Obtener la **clasificación** del tipo de tumor
   - Generar y visualizar la **máscara de segmentación**
   - Superponer la máscara sobre la imagen original en color rojo
   - Descargar los resultados en un archivo `.zip`

---

## 🛠️ Requisitos

- Python 3.8 o superior  
- [PyTorch](https://pytorch.org/) (CPU o GPU, según disponibilidad)  
- Paquetes detallados en el archivo `requirements_modelos.txt`  
- **Anaconda o Miniconda** para la correcta gestión del entorno (recomendado)

> 💡 **Nota**  
> El proyecto fue desarrollado utilizando **Anaconda**, por lo que se recomienda instalarlo antes de ejecutar la aplicación.

---

## 🚀 Instrucciones de uso (ejecución local)
> 📌 **Importante:** Todos los comandos deben ejecutarse desde  
> **Anaconda Prompt**, no desde CMD o PowerShell.  
> Esto asegura que `conda` funcione correctamente.

---

### 1. Clonar el repositorio

```bash
git clone https://github.com/Inkphemus-usm/PDI-TumorProcessing
cd PDI-TumorProcessing
```
### 2. Crear entorno conda
```bash
conda create -n pdi_tumores python=3.10
```
### 3. Activar el entorno
```bash
conda activate pdi_tumores
```
### 4. Instalar dependencias
```bash
pip install -r requirements_modelos.txt
```
### 5. Ir a la carpeta de la aplicación
```bash
cd Aplicacion
```
### 6. Ejecutar la aplicación Streamlit
```bash
streamlit run aplicacion.py
```
