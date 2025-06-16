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

# Preprocessing function for MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

def prepare_for_resnet(img):
    img = img.resize((100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array  # No preprocessing for ResNet

def prepare_for_mobilenet(img):
    img = img.resize((100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_preprocess(img_array)
    return img_array

def predict_weather(img):
    resnet_input = prepare_for_resnet(img)
    mobilenet_input = prepare_for_mobilenet(img)

    # Predict
    resnet_pred = resnet_model.predict(resnet_input)[0]
    mobilenet_pred = mobilenet_model.predict(mobilenet_input)[0]

    # Convert to {label: confidence}
    resnet_result = {class_labels[i]: float(resnet_pred[i]) for i in range(len(class_labels))}
    mobilenet_result = {class_labels[i]: float(mobilenet_pred[i]) for i in range(len(class_labels))}

    return resnet_result, mobilenet_result

# Gradio Interface
interface = gr.Interface(
    fn=predict_weather,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(label="ResNet50 Prediction"),
        gr.Label(label="MobileNet Prediction")
    ],
    title="Weather Phenomena Classifier",
    description="Upload an image and get predictions from ResNet50 and MobileNet."
)

interface.launch()
