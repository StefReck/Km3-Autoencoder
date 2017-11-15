# -*- coding: utf-8 -*-

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Flatten, Activation, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Cropping3D, Conv3DTranspose, AveragePooling3D
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import h5py

from compare_hists import *
from util.Loggers import *


def setup_simple_model():
    global model
    model=Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    
def setup_conv_model():
    #Test autoencoder im sequential style
    global conv_model
    conv_model = Sequential()
    #INput: 11x13x18 x 1
    
    conv_model.add(Conv3D(filters=16, kernel_size=(2,2,3), padding='valid', activation='relu', input_shape=(11, 13, 18, 1)))
    #10x12x16 x 16
    conv_model.add(AveragePooling3D((2, 2, 2), padding='valid'))
    #5x6x8 x 16
    conv_model.add(Conv3D(filters=8, kernel_size=(3,3,3), padding='valid', activation='relu' ))
    #3x4x6 x 8
    conv_model.add(Conv3D(filters=4, kernel_size=(2,3,3), padding='valid', activation='relu' ))
    #2x2x4 x 4

    #2x2x4 x 4
    conv_model.add(Conv3DTranspose(filters=8, kernel_size=(2,3,3), padding='valid', activation='relu' ))
    #3x4x6 x 8
    conv_model.add(Conv3DTranspose(filters=16, kernel_size=(3,3,3), padding='valid', activation='relu' ))
    #5x6x8 x 16
    conv_model.add(UpSampling3D((2, 2, 2)))
    #10x12x16 x 16
    conv_model.add(Conv3DTranspose(filters=1, kernel_size=(2,2,3), padding='valid', activation='relu' ))
    
    #Output 11x13x18 x 1
    conv_model.compile(optimizer='adadelta', loss='mse')
    
    #conv_model.summary()

def setup_conv_model_API():
    #Wie der autoencoder im sequential style, nur mit API
    global autoencoder
    
    inputs = Input(shape=(11,13,18,1))
    x = Conv3D(filters=16, kernel_size=(2,2,3), padding='valid', activation='relu')(inputs)
    #10x12x16 x 16
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 16
    x = Conv3D(filters=8, kernel_size=(3,3,3), padding='valid', activation='relu' )(x)
    #3x4x6 x 8
    encoded = Conv3D(filters=4, kernel_size=(2,3,3), padding='valid', activation='relu' )(x)
    #2x2x4 x 4
    
    
    #2x2x4 x 4
    x = Conv3DTranspose(filters=8, kernel_size=(2,3,3), padding='valid', activation='relu' )(encoded)
    #3x4x6 x 8
    x = Conv3DTranspose(filters=16, kernel_size=(3,3,3), padding='valid', activation='relu' )(x)
    #5x6x8 x 16
    x = UpSampling3D((2, 2, 2))(x)
    #10x12x16 x 16
    decoded = Conv3DTranspose(filters=1, kernel_size=(2,2,3), padding='valid', activation='relu' )(x)
    #Output 11x13x18 x 1
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')

def encoded_conv_model_API():
    #Erste Hälfte des conv_model_API autoencoder, um zu testen, wie import funktioniert.
    
    inputs = Input(shape=(11,13,18,1))
    x = Conv3D(filters=16, kernel_size=(2,2,3), padding='valid', activation='relu', trainable=False)(inputs)
    #10x12x16 x 16
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 16
    x = Conv3D(filters=8, kernel_size=(3,3,3), padding='valid', activation='relu', trainable=False )(x)
    #3x4x6 x 8
    encoded = Conv3D(filters=4, kernel_size=(2,3,3), padding='valid', activation='relu', trainable=False )(x)
    #2x2x4 x 4

    autoencoder = Model(inputs, encoded)
    return autoencoder
    
    
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/"
train_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5"
test_data = "test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5"

file=h5py.File('Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5', 'r')
xyz_hists = np.array(file["x"]).reshape((3498,11,13,18,1))
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
xyz_labels = np.array(file["y"])    


#setup_simple_model()
#setup_conv_model()
#setup_conv_model_API()

#encoder=encoded_conv_model_API()
#encoder.load_weights('../models/trained_autoencoder_test_epoch1.h5', by_name=True)
#encoder.compile(optimizer='adam', loss='mse')


"""
with open('Logfile.txt', 'w') as text_file:
    Testlog = NBatchLogger_Recent(display=1, logfile=text_file)
    Testlog = NBatchLogger_Epoch(display=1, logfile=text_file)
    history = autoencoder.fit(xyz_hists[0:100], xyz_hists[0:100], verbose=1, callbacks=[Testlog], epochs=2, batch_size=10)
"""

def plot_history():
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#plot_history()

def compare_events(no, model):
    original = xyz_hists[no].reshape(11,13,18)
    prediction = model.predict_on_batch(xyz_hists[no:(no+1)]).reshape(11,13,18)
    loss = model.evaluate(x=xyz_hists[no:(no+1)], y=xyz_hists[no:(no+1)])
    print("Loss: ",loss)
    compare_hists(original, prediction)


def plot_some_comparisons(model):
    for i in range(5):
        compare_events(i,model)
    
#autoencoder=load_model("../models/trained_autoencoder_vgg_0_epoch3.h5")

#plot_some_comparisons(autoencoder)
#compare_events(0,autoencoder)


def setup_vgg_like(make_autoencoder, modelpath_and_name=None):
    #a vgg-like autoencoder, witht lots of convolutional layers
    #If make_autoencoder==False only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    inputs = Input(shape=(11,13,18,1))
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder)(inputs)
    x = Conv3D(filters=32, kernel_size=(2,2,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder)(x)
    #10x12x16 x 32
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 64
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder )(x)
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder )(x)
    x = Conv3D(filters=64, kernel_size=(2,3,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder )(x)
    #4x4x6 x 64
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x2x3 x 64

    if make_autoencoder == True:
        #The Decoder part:
        
        #2x2x3 x 64
        x = UpSampling3D((2, 2, 2))(encoded)
        #4x4x6 x 64
        x = Conv3DTranspose(filters=64, kernel_size=(2,3,3), padding='valid', activation='relu' )(x)
        #5x6x8 x 64
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal' )(x)
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal' )(x)
        x = UpSampling3D((2, 2, 2))(x)
        #10x12x16 x 64
        x = Conv3DTranspose(filters=32, kernel_size=(2,2,3), padding='valid', activation='relu' )(x)
        #11x13x18 x 32
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        decoded = Conv3DTranspose(filters=1, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    elif make_autoencoder == False:
        #Replacement for the decoder part for supervised training:
        
        encoder= Model(inputs=inputs, outputs=encoded)
        encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        nb_classes=1
        outputs = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
        
model = setup_vgg_like(make_autoencoder=False, modelpath_and_name="../models/trained_vgg_0_autoencoder_epoch3.h5")
model.compile(optimizer='adam', loss='mse')

"""
train_on_batch(self, x, y, class_weight=None, sample_weight=None)

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, X_test))
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

test_on_batch(self, x, y, sample_weight=None)
predict_on_batch(self, x)
"""

