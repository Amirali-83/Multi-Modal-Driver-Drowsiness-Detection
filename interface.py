import tensorflow as tf
import numpy as np
import cv2
import gradio as gr

# Load model
model = tf.keras.models.load_model("driver_drowsiness_inference.keras")
class_labels = ["Closed", "Open", "no_yawn", "yawn"]  # Same order as training

# Prediction function
def predict_image(img):
    # Convert PIL to OpenCV (RGB to BGR)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Resize exactly like training
    img_resized = cv2.resize(img_cv, (160,160), interpolation=cv2.INTER_AREA)

    # Normalize exactly like ImageDataGenerator
    img_array = img_resized.astype("float32") / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred) * 100

    # Add label to image
    label = f"{class_labels[pred_class]} ({confidence:.1f}%)"
    cv2.putText(img_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Driver Drowsiness Detection",
    description="Upload a driver image to detect if eyes are closed, open, yawning, or not yawning."
)

iface.launch()
