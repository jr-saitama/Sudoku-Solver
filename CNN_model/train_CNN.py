## importing the required libraries ##

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import h5py
import numpy as np
import keras.utils as kut
import cv2
import matplotlib.pyplot as plt

## get the data set from saved hdf file ##

with h5py.File('data.h5', 'r') as hdf:
    X_train = np.array(hdf.get('X_train'))
    Y_train = np.array(hdf.get('Y_train'))
    print(Y_train.shape)

Y_onehot = kut.to_categorical(Y_train)  ## converting the categories into one_hot list
print(Y_onehot[:10])

X = tf.keras.utils.normalize(X_train, axis=1)  ## normalizing the data set

## Convolution Neural Network ##

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape =X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64, activation=tf.nn.relu))

model.add(Dense(32, activation=tf.nn.relu))

model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(X, Y_onehot, epochs= 20, validation_split = 0.1)

## saving the model ##
model.save('num_reader.model')

# predictions = model.predict([X])
# print(predictions)
# print(np.argmax(predictions[4]))
## loading the model ##
model = tf.keras.models.load_model("num_reader.model")
# predictions = model.predict([X])
# print(predictions)
final_string = ""

## preprocessing the test image ##

def preprocess(filepath):
    IMG_SIZE = 25
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = tf.keras.utils.normalize(img_array, axis=1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap ='gray')
    plt.show()
    print(prediction)
    return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 1)

for square in range(81):
    prediction = np.argmax(model.predict([preprocess("<file path>")])) # predicting on test images
    final_string+=str(prediction) 
print(final_string)
print(len(final_string))