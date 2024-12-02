import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(224, 224)):
    """
    Resizes all images in the input directory and saves them to the output directory.

    Args:
        input_dir (str): Path to the directory containing original images.
        output_dir (str): Path to the directory where resized images will be saved.
        size (tuple): Target size for resizing (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image extensions
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img = img.resize(size)
            img.save(os.path.join(output_dir, filename))
            print(f"Resized and saved: {filename}")

# Example usage:
input_directory = "Raw_Image"
output_directory = "Prep_Images"
resize_images(input_directory, output_directory)
