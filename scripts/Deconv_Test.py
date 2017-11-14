# -*- coding: utf-8 -*-

from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Activation, Conv3D, Conv2D, Conv2DTranspose, Conv3DTranspose, MaxPooling2D, MaxPooling3D, AveragePooling3D, UpSampling3D, UpSampling2D, Input, Lambda
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

from util.custom_layers import MaxUnpooling3D
from compare_hists import reshape_3d_to_3d
import h5py


def make_3d_plots_xyz(hist_org, hist_pred, title1, title2, suptitle=None):
    #Plot original and predicted histogram side by side in one plot
    #input format: [x,y,z,val]

    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(121, projection='3d', aspect='equal')
    max_value1= np.amax(hist_org)
    min_value1 = np.amin(hist_org) #min value is usually 0, but who knows if the autoencoder screwed up
    fraction1=(hist_org[3]-min_value1)/max_value1
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=8*36*fraction1, rasterized=True)
    cbar1=fig.colorbar(plot1,fraction=0.046, pad=0.1)
    cbar1.set_label('Hits', rotation=270, labelpad=0.1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(title1)
    
    ax2 = fig.add_subplot(122, projection='3d', aspect='equal')
    max_value2=np.amax(hist_pred)
    min_value2=np.amin(hist_pred)
    fraction2=(hist_pred[3]-min_value2)/max_value2
    plot2 = ax2.scatter(hist_pred[0],hist_pred[1],hist_pred[2], c=hist_pred[3], s=8*36*fraction2, rasterized=True)
    cbar2=fig.colorbar(plot2,fraction=0.046, pad=0.1)
    cbar2.set_label('Hits', rotation=270, labelpad=0.1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(title2)
    
    if suptitle is not None: fig.suptitle(suptitle)

    fig.tight_layout()

def compare_hists_xyz(hist_org, hist_pred, name, suptitle=None):
    make_3d_plots_xyz(reshape_3d_to_3d(hist_org), reshape_3d_to_3d(hist_pred), title1="MaxPooling", title2="AveragePooling", suptitle=suptitle)
    plt.savefig(name) 
    plt.close()


#2D Model
"""
inputs=Input(shape=(2,2,1))
x = Lambda(MaxUnpooling2D,MaxUnpooling2D_output_shape)(inputs)
model = Model(inputs=inputs, outputs=x)

mat=np.linspace(1,8,8).reshape((2,2,2,1))
res=model.predict(mat)
print(res)


"""

test_file = 'Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5'
file=h5py.File(test_file , 'r')
which=[3]
hists = file["x"][which].reshape((1,11,13,18,1))
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
labels=file["y"][which]



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


res=model.predict(hists)

get_layer_1_output = K.function([model.layers[0].input], [model.layers[1].output])
layer_1_output = get_layer_1_output([hists])[0]
get_layer_2_output = K.function([model.layers[0].input], [model.layers[2].output])
layer_2_output = get_layer_2_output([hists])[0]
get_layer_3_output = K.function([model.layers[0].input], [model.layers[3].output])
layer_3_output = get_layer_3_output([hists])[0]
get_layer_4_output = K.function([model.layers[0].input], [model.layers[4].output])
layer_4_output = get_layer_4_output([hists])[0]


get_layer_1_output_2 = K.function([model2.layers[0].input], [model2.layers[1].output])
layer_1_output_2 = get_layer_1_output_2([hists])[0]
get_layer_2_output_2 = K.function([model2.layers[0].input], [model2.layers[2].output])
layer_2_output_2 = get_layer_2_output_2([hists])[0]
get_layer_3_output_2 = K.function([model2.layers[0].input], [model2.layers[3].output])
layer_3_output_2 = get_layer_3_output_2([hists])[0]
get_layer_4_output_2 = K.function([model2.layers[0].input], [model2.layers[4].output])
layer_4_output_2 = get_layer_4_output_2([hists])[0]


#plot_hist(hists[0])
#plot_hist(layer_1_output[0])
#plot_hist(res[0])

compare_hists_xyz(hists[0], hists[0],name="comp0.pdf", suptitle="Input")
compare_hists_xyz(layer_1_output[0], layer_1_output_2[0], name="comp1.pdf", suptitle="One Pooling")
compare_hists_xyz(layer_2_output[0], layer_2_output_2[0], name="comp2.pdf",suptitle="Two Poolings")
compare_hists_xyz(layer_3_output[0], layer_3_output_2[0], name="comp3.pdf",suptitle="One UnPooling")
compare_hists_xyz(layer_4_output[0], layer_4_output_2[0], name="comp4.pdf",suptitle="Two UnPoolings")

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