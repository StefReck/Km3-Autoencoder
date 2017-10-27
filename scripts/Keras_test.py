# -*- coding: utf-8 -*-

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Cropping3D, Conv3DTranspose, AveragePooling3D
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import h5py
from compare_hists import *
from os import fsync

""" 
    on batch end params:
do_validation
verbose
metrics
batch_size
steps
epochs
samples
{'do_validation': True, 'verbose': 0, 'metrics': ['loss', 'val_loss'], 'batch_size': 10, 'steps': None, 'epochs': 2, 'samples': 270}

    logs
{'batch': 1, 'size': 10, 'loss': 1.9699528}
"""

class NBatchLogger_Recent(Callback):
    #Gibt lOSS aus alle :display batches, gemittelt über die letzten :display batches
    def __init__(self, display):
        self.seen = 0
        self.display = display
        self.average = 0

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        self.average += logs.get('loss')
        if self.seen % (self.display*logs.get('size', 0)) == 0:
            averaged_loss = self.average / (self.display)
            print ('\nSample {0}/{1} - Batch Loss: {2}'.format(self.seen,self.params['samples'], averaged_loss))
            self.average = 0
            
class NBatchLogger_Epoch(Callback):
    #Gibt lOSS aus alle display batches, gemittelt über alle batches dieser epoche (das ist das was unter der verbose bar steht)
    def __init__(self, display, logfile):
        self.seen = 0
        self.display = display
        self.average = 0
        self.logfile = logfile

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        self.average += logs.get('loss')
        if self.seen % (self.display*logs.get('size', 0)) == 0:
            averaged_loss = self.average * logs.get('size', 0) / (self.seen)
            print ('\nSample {0}/{1} - Batch Loss: {2}'.format(self.seen,self.params['samples'], averaged_loss))
            self.logfile.write('\nSample {0}/{1} - Batch Loss: {2}'.format(self.seen,self.params['samples'], averaged_loss))
            self.logfile.flush()
            fsync(self.logfile.fileno())
            
    def on_epoch_begin(self, epoch, logs={}):
        print ("Epoch",epoch)

    def on_epoch_end(self, epoch, logs={}):
        self.seen = 0
        self.average  = 0

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

    
    
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/"
train_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5"
test_data = "test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5"

file=h5py.File('Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5', 'r')
xyz_hists = np.array(file["x"]).reshape((3498,11,13,18,1))
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
xyz_labels = np.array(file["y"])    


#setup_simple_model()
#setup_conv_model()
setup_conv_model_API()

# Output batch loss every n batches
#out_batch = NBatchLogger(display=10)


with open('Logfile.txt', 'w') as text_file:

    Testlog = NBatchLogger_Recent(display=1)
    Testlog2 = NBatchLogger_Epoch(display=2, logfile=text_file)

    history = autoencoder.fit(xyz_hists[0:100], xyz_hists[0:100], verbose=0, callbacks=[Testlog2], epochs=3, batch_size=10)


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
    compare_hists(original, prediction)

"""
train_on_batch(self, x, y, class_weight=None, sample_weight=None)

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, X_test))
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

test_on_batch(self, x, y, sample_weight=None)
predict_on_batch(self, x)
"""

