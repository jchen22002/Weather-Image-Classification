import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
resnet_model = load_model('resnet50_model.h5')
mobilenet_model = load_model('mobilenet_model.h5')

# Class labels
class_labels = ['dew', 'fog/smog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

def prepare_image(img):
    """Preprocess the image for prediction"""
    img = img.resize((100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_weather(img):
    img_array = prepare_image(img)

    # Predict
    resnet_pred = resnet_model.predict(img_array)
    mobilenet_pred = mobilenet_model.predict(img_array)

    resnet_result = class_labels[np.argmax(resnet_pred)]
    mobilenet_result = class_labels[np.argmax(mobilenet_pred)]

    return {
        "ResNet50 Prediction": resnet_result,
        "MobileNet Prediction": mobilenet_result
    }

# Build Gradio Interface
interface = gr.Interface(
    fn=predict_weather,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Weather Phenomena Classifier",
    description="Upload an image and get predictions from ResNet50 and MobileNet."
).launch()