# utils.py
from PIL import Image, ImageOps
import numpy as np

def preprocess_image(uploaded_image):
    # Convert to grayscale, invert, resize to 28x28
    img = Image.open(uploaded_image).convert("L")  # grayscale
    img = ImageOps.invert(img)                     # make digits white
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1)
    img_array = img_array.astype("float32") / 255.0
    return img_array
