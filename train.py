import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os

# Define the paths to the training data
train_data_dir = 'BananaLSD/AugmentedSet'

# Function to load images and labels from a folder
def load_data_from_folder(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = Image.open(img_path).resize((224, 224))
                img = np.array(img)
                images.append(img)
                labels.append(label_folder)  # Assuming folder name is the label
    return np.array(images), np.array(labels)

# Load the training data
train_images, train_labels = load_data_from_folder(train_data_dir)

# Convert labels to numerical values
label_to_index = {label: idx for idx, label in enumerate(set(train_labels))}
train_labels = np.array([label_to_index[label] for label in train_labels])

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Preprocess images to match the input format of InceptionV3
train_images = tf.keras.applications.inception_v3.preprocess_input(train_images)
val_images = tf.keras.applications.inception_v3.preprocess_input(val_images)

# Convert labels to categorical if needed (for classification tasks)
train_labels = tf.keras.utils.to_categorical(train_labels)
val_labels = tf.keras.utils.to_categorical(val_labels)

# Load the InceptionV3 model pre-trained on ImageNet, exclude the top layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model (optional, depending on if you want to fine-tune)
base_model.trainable = False

# Add custom layers on top for our classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling to reduce dimensionality
x = Dense(1024, activation='relu')(x)  # Dense layer
predictions = Dense(len(label_to_index), activation='softmax')(x)  # Output layer (number of classes)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=32)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Print all training and validation metrics
print("Training and validation metrics:")
for key in history.history.keys():
    print(f"{key}: {history.history[key]}")
