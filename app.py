from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
resnet_model = load_model('resnet50_model.h5')
mobilenet_model = load_model('mobilenet_model.h5')

# Define the class labels 
class_labels = ['dew', 'fog/smog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

def prepare_image(img_path):
    """Preprocess the image for prediction"""
    img = image.load_img(img_path, target_size=(100, 100))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded file
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)
    
    # Prepare image for prediction
    img_array = prepare_image(img_path)
    
    # Predict using both models (ResNet and MobileNet)
    resnet_prediction = resnet_model.predict(img_array)
    mobilenet_prediction = mobilenet_model.predict(img_array)

    # Get predicted class labels
    resnet_class = class_labels[np.argmax(resnet_prediction)]
    mobilenet_class = class_labels[np.argmax(mobilenet_prediction)]
    
    return render_template('result.html', resnet_class=resnet_class, mobilenet_class=mobilenet_class)

if __name__ == "__main__":
    app.run(debug=True)
