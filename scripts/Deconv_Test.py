# -*- coding: utf-8 -*-

from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv3D, Conv2D, Conv2DTranspose, Conv3DTranspose
import numpy as np
import matplotlib.pyplot as plt


model = Sequential()
#model.add(Conv2D(filters=1, kernel_size=(2,2), padding='valid', activation='relu', kernel_initializer=initializers.Constant(value=1), input_shape=(3, 3, 1)))
model.add(Conv3DTranspose(filters=1, kernel_size=(2,2,2), padding='valid', kernel_initializer=initializers.Constant(value=1), activation='relu', input_shape=(2,2,2,1)))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])


inp = np.ones((1,2,2,2,1))

prediction = model.predict_on_batch(inp).astype(int)

def display():
    print(inp.reshape(2,2,2))
    print(prediction.reshape(3,3,3))

