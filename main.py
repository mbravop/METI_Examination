import streamlit as st
import numpy as np
from PIL import Image

st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    st.write(f"Generated images of digit {digit}")

    # Esta sección generará imágenes con tu modelo más adelante.
    # Por ahora, generará placeholders aleatorios:
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        random_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        image = Image.fromarray(random_image, 'L')
        
        col.image(image, caption=f"Sample {idx+1}")
