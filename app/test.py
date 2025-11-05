# =====================================================
# Coral Reef Health Classification - Single Image Test
# =====================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import os

# Configuration
IMG_WIDTH, IMG_HEIGHT = 224, 224
MODEL_PATH = 'coral_reef_resnet50_model.h5'
CLASS_INDICES_PATH = 'class_indices.pkl'

# -----------------------------------------------------
# Load Model and Class Labels
# -----------------------------------------------------
def load_model_and_classes():
    """Load the trained model and class indices"""
    model = keras.models.load_model(MODEL_PATH)

    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)

    # Reverse mapping (index → class name)
    class_names = {v: k for k, v in class_indices.items()}
    return model, class_names


# -----------------------------------------------------
# Predict a Single Image
# -----------------------------------------------------
def predict_single_image(img_path, model, class_names):
    """Predict coral reef health for one image"""
    # Load and preprocess image
    img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    
    # For binary classification (Healthy vs Bleached)
    pred_class = int(prediction[0] > 0.5)
    confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]
    predicted_label = class_names[pred_class]

    return predicted_label, confidence


# -----------------------------------------------------
# Generate GradCAM Heatmap
# -----------------------------------------------------
def generate_gradcam(img_path, model, class_names, save_path="gradcam_result.png"):
    import tensorflow as tf
    import numpy as np
    import cv2

    # --- Load and preprocess image ---
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Get prediction from top-level model ---
    prediction = model.predict(img_array, verbose=0)
    pred_class = int(prediction[0][0] > 0.5)
    confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]
    predicted_label = class_names[pred_class]
    print(f"Prediction: {predicted_label}, Confidence: {confidence:.2%}")

    # --- Find base model (ResNet50) ---
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    if base_model is None:
        base_model = model

    # --- Find last Conv2D layer ---
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the model.")

    # --- Create Grad-CAM model using only the base model ---
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # --- Compute feature map and gradients ---
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs = grad_model(inputs)
        # For binary classification, take mean of last conv outputs for Grad-CAM
        loss = tf.reduce_mean(conv_outputs)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # --- Normalize heatmap ---
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    # --- Overlay heatmap on original image ---
    img_orig = cv2.imread(img_path)
    img_orig = cv2.resize(img_orig, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

    # --- Save and display ---
    cv2.imwrite(save_path, superimposed_img)
    print(f"Grad-CAM saved at: {save_path}")
    cv2.imshow("Grad-CAM", superimposed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------------------------------
# Main Function
# -----------------------------------------------------
if __name__ == "__main__":
    print("Loading model...")
    model, class_names = load_model_and_classes()
    print("Model loaded successfully!")
    print(f"Class mapping: {class_names}")

    # Example: specify your test image path here
    test_image_path = "coral1_jpg.rf.f3e4be06deaef9b5f01d01a32a8dc6b5.jpg"  # <-- Replace this with your image path

    if os.path.exists(test_image_path):
        print(f"\nTesting image: {test_image_path}")
        predicted_label, confidence = predict_single_image(test_image_path, model, class_names)
        print(f"\nPrediction: {predicted_label}")
        print(f"Confidence: {confidence:.2%}")

        # Generate GradCAM visualization
        generate_gradcam(test_image_path, model, class_names)
    else:
        print(f"❌ Image not found: {test_image_path}")
        print("Please provide a valid image path.")
