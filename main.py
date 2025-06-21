import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Parámetros
latent_dim = 100
num_classes = 10

# Cargar el modelo una sola vez
@st.cache_resource
def load_generator_model():
    return tf.keras.models.load_model("modelo/modeloFinal.keras")

generator = load_generator_model()

# Interfaz de usuario
st.title("🧠 Generador de numeros (0–9)")
digit = st.selectbox("Selecciona un dígito:", list(range(10)))
generate = st.button("Generar 5 imágenes")

if generate:
    st.subheader(f"Imágenes del dígito {digit}:")
    cols = st.columns(5)

    # Generar ruido + etiqueta codificada one-hot
    noise = np.random.normal(0, 1, (5, latent_dim))
    labels = tf.keras.utils.to_categorical([digit] * 5, num_classes=num_classes).astype(np.float32)

    # Generar imágenes
    generated_images = generator.predict([noise, labels])

    for i in range(5):
        img_array = generated_images[i, :, :, 0]
        img_array = ((img_array + 1) * 127.5).astype(np.uint8)  # Escala [-1,1] → [0,255]
        img = Image.fromarray(img_array, mode='L')
        cols[i].image(img, caption=f"{digit} #{i+1}")
