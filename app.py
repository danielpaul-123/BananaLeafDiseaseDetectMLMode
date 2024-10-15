from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Define the class labels for your model's predictions
class_labels = ['Healthy', 'Banana Black Sigatoka Disease', 'Banana Bract Mosaic Virus Disease', 'Banana Insect Pest Disease','Banana Moko Disease','Banana Panama Disease','Banana Yellow Sigatoka Disease']  # Update with your actual class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/healthy')
def healthy():
    return render_template('0.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)

        # Preprocess the image for the model
        img = image.load_img(filepath, target_size=(299, 299))  # Adjust target size based on your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Load your pre-trained model
        model = load_model('banana_leaf.h5')

        # Run the model
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        
        return f'Predicted class: {predicted_class_label}'

        clear_model()

if __name__ == "__main__":
    app.run(debug=True)
