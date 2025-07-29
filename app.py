import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import load_model
from PIL import Image


# ✅ Load model from Google Drive mount path or local fallback
MODEL_PATH = "symbol_classifier.h5"  # or Google Drive path if mounted

# === Load Model ===
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# === Class labels (same order as training) ===
CLASS_NAMES = sorted(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'add', 'dec', 'div', 'eq', 'mul', 'sub', 'x', 'y', 'z'])

# === Image preprocessing ===
def preprocess_image(image):
    img = np.array(image.convert("L"))
    img = cv2.resize(img, (45, 45))
    img = img / 255.0
    img = img.reshape(1, 45, 45, 1)
    return img

# === Streamlit UI ===
st.title("🧠 Math Symbol Recognizer")
# === Welcome Message with Instructions ===
st.markdown("""
Welcome! Upload an image of a **handwritten math symbol** (like `+`, `x`, `y`, `1`, etc.) and let the AI recognize it.

⚠️ **For best results:**
- Use **black ink** or **dark marker**
- Write on a **clean white background**
- Make sure the symbol is **centered and clearly visible**
""")
uploaded_file = st.file_uploader("Upload handwritten symbol image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)

    try:
        img = preprocess_image(image)
        prediction = model.predict(img)
        predicted_label = CLASS_NAMES[np.argmax(prediction)]

        st.success(f"✅ Predicted Symbol: **{predicted_label}**")
        st.bar_chart(prediction[0])
    except Exception as e:
        st.error(f"Error: {e}")
