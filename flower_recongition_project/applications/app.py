import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('../final_model/Flower_Recog_Model.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader(label='Upload Images', accept_multiple_files=True)
num_columns = 4
columns = st.columns(num_columns)
if uploaded_file is not None:
    for i,image in enumerate(uploaded_file):
        with open(os.path.join(os.getcwd(),'../upload', image.name), 'wb') as f:
            f.write(image.getbuffer())
    
        columns[i % num_columns].image(image, width = 80, use_column_width=True)
    
        columns[i % num_columns].markdown(classify_images(image))

