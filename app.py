# app.py
import streamlit as st
import tensorflow as tf
from utils import preprocess_image
import numpy as np

# Load model
model = tf.keras.models.load_model("mnist_cnn.h5")

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🔢 Handwritten Digit Recognition")
st.markdown("Upload an image (28x28 or larger) of a digit (0–9).")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=150)

    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: **{digit}**")
