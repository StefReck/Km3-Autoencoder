# -*- coding: utf-8 -*-

from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv3D, Conv2D, Conv2DTranspose, Conv3DTranspose
import numpy as np
import matplotlib.pyplot as plt



inputs = Input(shape=(3,3,1))
x = Conv2D(filters=2, kernel_size=(2,2), padding='valid', activation='relu', kernel_initializer=initializers.Constant(value=1))(inputs)
y = Conv2DTranspose(filters=1, kernel_size=(2,2), padding='valid', kernel_initializer=initializers.Constant(value=-1), activation='relu')(x)
model = Model(inputs, y)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])


inp = np.ones((1,3,3,1))

prediction = model.predict_on_batch(inp).astype(int)

def display():
    print(inp.reshape(2,2,2))
    print(prediction.reshape(3,3,3))

weights=[]
for layer in model.layers:
    weights.append(layer.get_weights())
    

model.save('test.h5')
del model


inputs2 = Input(shape=(3,3,1))
x2 = Conv2D(filters=2, kernel_size=(2,2), padding='valid', activation='relu', kernel_initializer=initializers.Constant(value=1))(inputs2)
encoder = Model(inputs2, x2)
encoder.load_weights('test.h5', by_name=True)

weights2=[]
for layer in encoder.layers:
    weights2.append(layer.get_weights())