# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Cropping3D, Conv3DTranspose, AveragePooling3D
import numpy as np
import matplotlib.pyplot as plt
import h5py
from compare_hists import *

def setup_simple_model():
    global model
    model=Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    
def setup_conv_model():
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

    
#setup_simple_model()
#setup_bug_model()
setup_conv_model()

file=h5py.File('Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5', 'r')
xyz_hists = np.array(file["x"]).reshape((3498,11,13,18,1))

# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
xyz_labels = np.array(file["y"])



history = conv_model.fit(xyz_hists[0:300], xyz_hists[0:300], epochs=4, batch_size=20, validation_split=0.1)

def plot_history():
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#plot_history()

def compare_events(no):
    original = xyz_hists[no].reshape(11,13,18)
    prediction = conv_model.predict_on_batch(xyz_hists[no:(no+1)]).reshape(11,13,18)
    compare_hists(original, prediction)

"""
train_on_batch(self, x, y, class_weight=None, sample_weight=None)

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, X_test))
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

test_on_batch(self, x, y, sample_weight=None)
predict_on_batch(self, x)
"""

