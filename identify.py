import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import io

model = load_model('banana_leaf.h5')

# Function to preprocess the input image for prediction
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).resize((299, 299))  # Resize to 299x299
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)  # Preprocess for InceptionV3
    return img_array

# 9. Load and preprocess a new image for prediction
img_path = 'test_images/test_2.jpeg'  # Path to the image you want to classify
preprocessed_image = load_and_preprocess_image(img_path)

# 10. Make predictions
predictions = model.predict(preprocessed_image)

# 11. Interpret the results
predicted_class_index = np.argmax(predictions, axis=1)

# Assuming you have a mapping of class indices to class names
class_labels = ['Healthy', 'Black Sigatoka', 'Bract Mosaic Virus','Insect Pest','Moko','Panama','Yellow Sigatoka']  # Replace with your actual class names
predicted_class_label = class_labels[predicted_class_index[0]]

print(f"Predicted class index: {predicted_class_index[0]}")
print(f"Predicted class label: {predicted_class_label}")
