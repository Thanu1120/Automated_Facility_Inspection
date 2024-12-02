import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def test_model(model_path, image_path, class_indices):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class = class_names[np.argmax(predictions)]

    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    model_path = "facility_inspection_model.h5"
    test_image_path = "Testimage4.jpeg"  # Replace with your image path
    class_indices = {'good': 0, 'damaged': 1, 'needs_repair': 2}  # Update as per training
    
    test_model(model_path, test_image_path, class_indices)
