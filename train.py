import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import io

# 1. Load the Parquet file and extract images and labels
train_data = pd.read_parquet('train-00000-of-00001.parquet')
test_data = pd.read_parquet('test-00000-of-00001.parquet')

# Assuming 'image' column contains image bytes and 'label' column contains labels
train_images = []
train_labels = train_data['label'].values

test_images = []
test_labels = test_data['label'].values


for img_data in train_data['image']:
    # Convert image bytes to a PIL image and resize to 299x299 (InceptionV3 input size)
    image_bytes = img_data['bytes']  # Use the actual key for image bytes
    img = Image.open(io.BytesIO(image_bytes)).resize((299, 299))
    img = np.array(img)
    train_images.append(img)

for img_data in test_data['image']:
    # Convert image bytes to a PIL image and resize to 299x299 (InceptionV3 input size)
    image_bytes = img_data['bytes']  # Use the actual key for image bytes
    img = Image.open(io.BytesIO(image_bytes)).resize((299, 299))
    img = np.array(img)
    test_images.append(img)

# Convert images to a NumPy array
train_images = np.array(train_images)
test_images = np.array(test_images)

# Preprocess images to match the input format of InceptionV3
train_images = tf.keras.applications.inception_v3.preprocess_input(train_images)
test_images = tf.keras.applications.inception_v3.preprocess_input(test_images)

# Convert labels to categorical if needed (for classification tasks)
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# 3. Load the InceptionV3 model pre-trained on ImageNet, exclude the top layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze the base model (optional, depending on if you want to fine-tune)
base_model.trainable = False

# 4. Add custom layers on top for our classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling to reduce dimensionality
x = Dense(1024, activation='relu')(x)  # Dense layer
predictions = Dense(train_labels.shape[1], activation='softmax')(x)  # Output layer (number of classes)

# 5. Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train the model
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=32)

# 8. Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")
