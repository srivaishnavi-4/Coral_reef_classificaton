import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
import pickle
import cv2
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
IMG_WIDTH, IMG_HEIGHT = 224, 224
MODEL_PATH = 'coral_reef_resnet50_model.h5'
CLASS_INDICES_PATH = 'class_indices.pkl'

# -----------------------------
# Load Model & Class Labels
# -----------------------------
@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    return model, class_names

# -----------------------------
# Predict Function
# -----------------------------
def predict_single_image(img_array, model, class_names):
    prediction = model.predict(img_array, verbose=0)
    pred_class = int(prediction[0] > 0.5)
    confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]
    predicted_label = class_names[pred_class]
    return predicted_label, confidence

# -----------------------------
# Grad-CAM Function
# -----------------------------
def generate_gradcam(img_array, model):
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    if base_model is None:
        base_model = model

    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found!")

    grad_model = tf.keras.models.Model(inputs=base_model.input, outputs=last_conv_layer.output)
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_tensor)
        loss = tf.reduce_mean(conv_outputs)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10
    heatmap = cv2.resize(heatmap.numpy(), (IMG_WIDTH, IMG_HEIGHT))
    return heatmap

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🐠 Coral Reef Health Classification", layout="wide")
st.title("🐠 Coral Reef Health Classification")

model, class_names = load_model_and_classes()

# Columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Resize to fit column
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("Prediction & Confidence")
    if uploaded_file is not None:
        img_array = img_to_array(image)
        img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        input_array = np.expand_dims(img_array / 255.0, axis=0)

        predicted_label, confidence = predict_single_image(input_array, model, class_names)
        st.markdown(f"**Class:** {predicted_label}")
        st.markdown(f"**Confidence:** {confidence:.2%}")

        st.subheader("Grad-CAM")
        heatmap = generate_gradcam(input_array, model)
        img_orig = np.array(image.resize((IMG_WIDTH, IMG_HEIGHT)))
        superimposed_img = overlay_heatmap(img_orig, heatmap)
        st.image(superimposed_img, caption="Grad-CAM Overlay", use_column_width=True)
