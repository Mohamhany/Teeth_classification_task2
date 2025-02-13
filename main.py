import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array 
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('teeth_classification_pretrain (1).h5')
label_map = {
    0: "CaS",
    1: "CoS",
    2: "Gum",
    3:"MC",
    4:"OC",
    5:"OLP",
    6:"OT"
    }

# Initialize ImageDataGenerator for preprocessing
datagen = ImageDataGenerator()

# Streamlit app layout
st.title("teeth Classification")
st.write("Upload an image to classify")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)  
    image = datagen.flow(image, batch_size=1,).__next__()  

    # Predict using the model
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_label = label_map.get(predicted_class_index)

    # Display prediction
    st.write("Predicted Label:", predicted_label)

