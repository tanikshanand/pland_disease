import os  # Filesystem
import json  # JSON handling
from PIL import Image  # Image processing

import numpy as np  # Numeric operations
import tensorflow as tf  # Deep learning
import streamlit as st  # Web interface


working_dir = os.path.dirname(os.path.abspath(__file__))  # Directory
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"  # Model path

model = tf.keras.models.load_model(model_path)  # Load model

class_indices = json.load(open(f"{working_dir}/class_indices.json"))  # Class indices


def load_and_preprocess_image(image_path, target_size=(224, 224)):  # Preprocessing
    
    img = Image.open(image_path)  # Open image
    img = img.resize(target_size)  # Resize image
    img_array = np.array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array = img_array.astype('float32') / 255.  # Normalize
    return img_array  # Return image array


def predict_image_class(model, image_path, class_indices):  # Prediction
    preprocessed_img = load_and_preprocess_image(image_path)  # Preprocess
    predictions = model.predict(preprocessed_img)  # Make prediction
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get predicted index
