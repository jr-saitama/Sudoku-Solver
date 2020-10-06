## importing the required libraries ##

import tensorflow as tf
from tensorflow.keras.models import Sequential
import h5py
import numpy as np
import keras.utils as kut
import cv2
import matplotlib.pyplot as plt
import os

def preprocess(filename, IMG_SIZE= 25):
    img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img_array = tf.keras.utils.normalize(img_array, axis=1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # plt.imshow(new_array, cmap ='gray')
    # plt.show()
    # print(prediction)
    return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 1)

def predict(Filepath):
    final_string = ""
    path = os.getcwd()
    path = os.path.join(os.path.dirname(path), "CNN_model")
    path = os.path.join(path, "num_reader.model")
    model = tf.keras.models.load_model(path)
    	
    for square in range(81):
    	prediction = np.argmax(model.predict([preprocess(Filepath+str(square)+".png")])) # predicting on test images
    	final_string+=str(prediction) 
    return final_string