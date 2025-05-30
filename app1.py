import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load models
detection_model = YOLO("yolov8n.pt")   # for person detection
classification_model = load_model("MobileNetV2_15.h5")  # for uniform classification

# Preprocessing function using PIL
def preprocess_image(image):
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.array(image)
    image = image / 255.0
    image = np.reshape(image, [1, 512, 512, 3])
    return image

# Prediction function
def predict_image(img):
    pred = classification_model.predict(img, verbose=0)
    return 1 if pred[0][0] >= 0.5 else 0, pred[0][0]

# Streamlit App UI
st.set_page_config(page_title="College Uniform Classifier", layout="wide")
st.title("College Uniform Classification")
st.write("Upload an image or capture a photo. We will first detect the person, then predict if wearing college uniform.")

# Image input section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
captured_file = st.camera_input("Capture an Image")

if uploaded_file or captured_file:
    file = uploaded_file if uploaded_file else captured_file
    pil_image = Image.open(file).convert('RGB')
    image_array = np.array(pil_image)

    # Run detection model
    results = detection_model(image_array)

    draw = ImageDraw.Draw(pil_image)
    found_person = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if class_id == 0:  # Class 0 means 'person'
                found_person = True

                # Crop the person
                person_crop = pil_image.crop((x1, y1, x2, y2))

                # Preprocess and predict
                processed_crop = preprocess_image(person_crop)
                prediction, probability = predict_image(processed_crop)

                # Set label and color
                label = f"Wearing Uniform: {probability:.2f}" if prediction == 1 else f"Not Wearing Uniform: {probability:.2f}"
                color = (0, 255, 0) if prediction == 1 else (255, 0, 0)

                # Draw bounding box and label
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                font = ImageFont.truetype("arial.ttf", size=20) 
                text_size = draw.textsize(label, font=font)
                draw.rectangle([ (x1, y1 - text_size[1] - 4), (x1 + text_size[0] + 4, y1) ], fill=color)
                draw.text((x1 + 2, y1 - text_size[1] - 2), label, fill='white', font=font)

    # Display results
    st.image(pil_image, caption="Processed Image", use_container_width=True)

    if not found_person:
        st.warning("No students detected in the image!")

# Style enhancements
st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f7f9fc;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)
