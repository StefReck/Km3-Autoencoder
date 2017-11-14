# -*- coding: utf-8 -*-

from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Activation, Conv3D, Conv2D, Conv2DTranspose, Conv3DTranspose, MaxPooling2D, MaxPooling3D, AveragePooling3D, UpSampling3D, UpSampling2D, Input, Lambda
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

from util.custom_layers import MaxUnpooling3D
from compare_hists import *
import h5py




#2D Model
"""
inputs=Input(shape=(2,2,1))
x = Lambda(MaxUnpooling2D,MaxUnpooling2D_output_shape)(inputs)
model = Model(inputs=inputs, outputs=x)

mat=np.linspace(1,8,8).reshape((2,2,2,1))
res=model.predict(mat)
print(res)


"""
inputs=Input(shape=(11,13,18,1))

x = MaxPooling3D(padding="same")(inputs)
x = MaxPooling3D(padding="same")(x)
x = MaxUnpooling3D(x,(2,2,2))
out = MaxUnpooling3D(x,(2,2,2))
model = Model(inputs=inputs, outputs=out)

x2 = AveragePooling3D(padding="same")(inputs)
x2 = AveragePooling3D(padding="same")(x2)
x2 = UpSampling3D((2,2,2))(x2)
out2 = UpSampling3D((2,2,2))(x2)
model2 = Model(inputs=inputs, outputs=out2)

x3 = AveragePooling3D()(inputs)
out3 = MaxUnpooling3D(x3,(2,2,2))
model3 = Model(inputs=inputs, outputs=out3)


test_file = 'Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5'
file=h5py.File(test_file , 'r')
which=[3]
hists = file["x"][which].reshape((1,11,13,18,1))
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
labels=file["y"][which]

res=model.predict(hists)

get_layer_1_output = K.function([model.layers[0].input], [model.layers[1].output])
layer_1_output = get_layer_1_output([hists])[0]
get_layer_2_output = K.function([model.layers[0].input], [model.layers[2].output])
layer_2_output = get_layer_2_output([hists])[0]

get_layer_1_output_2 = K.function([model2.layers[0].input], [model2.layers[1].output])
layer_1_output_2 = get_layer_1_output_2([hists])[0]

#plot_hist(hists[0])
#plot_hist(layer_1_output[0])
#plot_hist(res[0])

compare_hists_xzt(hists[0], res[0], suptitle="Max Pooling")
compare_hists_xzt(hists[0], model2.predict(hists)[0], suptitle="Average Pooling")
plot_hist(layer_1_output[0])
plot_hist(layer_2_output[0])
#compare_hists_xzt(hists[0], model3.predict(hists)[0], suptitle="C Average Pooling")

#compare_hists_xzt(hists[0], layer_1_output[0], suptitle="Max Pooling Intermediate Layer")
#compare_hists_xzt(hists[0], layer_1_output_2[0], suptitle="Average Pooling Intermediate Layer")

#plot_hist(layer_1_output[0])
#plot_hist(layer_1_output_2[0])


"""
mat=np.linspace(1,8,8).reshape((1,2,2,2,1))
res=model.predict(mat)
print(res)



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