import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_facility_condition(model, img_array, class_indices):
    predictions = model.predict(img_array)
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

st.title("Automated Facility Inspection")
uploaded_file = st.file_uploader("Upload an image of the facility", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    model = tf.keras.models.load_model("facility_inspection_model.h5")
    class_indices = {'good': 0, 'damaged': 1, 'needs_repair': 2}  # Update as per training

    prediction = predict_facility_condition(model, img_array, class_indices)
    st.image(uploaded_file, caption=f"Predicted Condition: {prediction}")
