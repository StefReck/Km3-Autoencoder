# -*- coding: utf-8 -*-

from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv3D, Conv2D, Conv2DTranspose, Conv3DTranspose, MaxPooling2D, UpSampling2D, Input, Lambda
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


def MaxUnpooling2D(InputTensor):
    #[1,2],[3,4]
    shape=K.int_shape(InputTensor)
    #(2,2)
    length=shape[0] #4
    resh = K.reshape(InputTensor, (4,1 ))
    #
    padding = K.zeros((4,1))
    padded=K.concatenate((resh,padding), axis=1)
    
    out = K.reshape(padded, (2,4 ))
    
    empty=K.zeros((2,4))
    padded2=K.concatenate((out,empty), axis=0)
    
    return padded2

inputs=K.ones(shape=(2,2))
print("Input:\n",K.eval(inputs))
print("\nShape: ",K.eval(inputs).shape, "\n")
x = MaxUnpooling2D(inputs)
print("Output:\n",K.eval(x))
print("\nShape: ",K.eval(x).shape, "\n")

def out_shape(input_shape):
    shape=list(input_shape)
    shape[-1] *= 2
    return tuple(shape)

"""
inputs = Input(shape=(5,1))
x = Lambda(MaxUnpooling1D, output_shape=out_shape)(inputs)
model = Model(inputs, x)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
"""
"""
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
    
"""