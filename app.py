# # app.py
# import streamlit as st
# import tensorflow as tf
# from utils import preprocess_image
# import numpy as np

# # Load model
# model = tf.keras.models.load_model("mnist_cnn.h5")

# st.set_page_config(page_title="Digit Recognizer", layout="centered")
# st.title("🔢 Handwritten Digit Recognition")
# st.markdown("Upload an image (28x28 or larger) of a digit (0–9).")

# uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", width=150)

#     img_array = preprocess_image(uploaded_file)
#     prediction = model.predict(img_array)
#     digit = np.argmax(prediction)

#     st.success(f"✅ Predicted Digit: **{digit}**")
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
model = tf.keras.models.load_model('mnist_cnn.h5')

st.title("🖐️ Handwritten Digit Recognizer")
st.write("Upload a 28x28 PNG image with a white digit on black background.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale

    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess the image
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: **{digit}**")
