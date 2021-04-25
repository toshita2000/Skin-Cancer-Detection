# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:13:27 2021

@author: Toshita Sharma
"""

import streamlit as st
import os
from tensorflow.keras.preprocessing.image import load_img
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

from tensorflow.keras import preprocessing
import PIL
import numpy as np
import cv2


st.title("Skin Cancer Prediction")
st.sidebar.title("Skin Cancer Prediction")
st.sidebar.image("side.jpg")
st.sidebar.subheader("By:-")
st.sidebar.subheader("18BEC115 - Riya Tanna & 18BEC118 - Toshita Sharma")
st.image("main.jpg")
st.sidebar.subheader(" ")
st.write("The model is CNN trained with three layers and a ReLu activation function")
test = 'C:/Users/Toshita Sharma/Desktop/test'
validation_dir = os.path.join(test)
from PIL import Image, ImageOps
import numpy as np

basepath = os.path.dirname(__file__)
MODEL_PATH = 'model.h5'
file_path = os.path.join(basepath, MODEL_PATH)
model = load_model('model.h5')
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
image = Image.open(file)
st.image(image, width = 250)
    
image_size = 112
try:
        image = cv2.cvtColor(np.array(image),cv2.IMREAD_COLOR)
        image = cv2.resize(image,(image_size,image_size))
finally:
    print("done")
    val = img_to_array(image)
    vals = val.reshape(image_size,image_size,3)    
    arr = np.expand_dims(vals, axis=0)
    arr /= 255
    prediction = (model.predict(arr) > 0.5).astype("int32")
    pred = prediction[0]
    print((st.write('This skin mole is malignant')if pred >0 else st.write('This skin mole is benign')))
