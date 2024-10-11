from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load your pre-trained model
model = load_model('my_model.h5')

# Define the class labels for your model's predictions
class_labels = ['class_0', 'class_1', 'class_2', ...]  # Update with your actual class names

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Ensure the uploaded file is valid
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        return JSONResponse(status_code=400, content={"error": "Invalid file type"})

    try:
        # Read the image from the uploaded file
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).resize((299, 299))  # Resize for InceptionV3
        
        # Convert the image to a numpy array and preprocess it for the model
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        
        return JSONResponse({
            'predicted_class_index': int(predicted_class_index),
            'predicted_class_label': predicted_class_label
        })
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error processing image: {str(e)}"})

# Basic endpoint to ensure the server is running
@app.get("/")
def root():
    return {"message": "FastAPI model server is running"}
