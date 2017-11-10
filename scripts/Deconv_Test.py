# -*- coding: utf-8 -*-

from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Activation, Conv3D, Conv2D, Conv2DTranspose, Conv3DTranspose, MaxPooling2D, UpSampling2D, Input, Lambda
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

from util.custom_layers import MaxUnpooling3D





#2D Model
"""
inputs=Input(shape=(2,2,1))
x = Lambda(MaxUnpooling2D,MaxUnpooling2D_output_shape)(inputs)
model = Model(inputs=inputs, outputs=x)

mat=np.linspace(1,8,8).reshape((2,2,2,1))
res=model.predict(mat)
print(res)


"""
inputs=Input(shape=(2,2,2,1))
x = MaxUnpooling3D(inputs)
model = Model(inputs=inputs, outputs=x)

mat=np.linspace(1,8,8).reshape((1,2,2,2,1))
res=model.predict(mat)
print(res)


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