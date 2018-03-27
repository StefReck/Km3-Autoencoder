# -*- coding: utf-8 -*-
"""
C
"""
from keras.models import load_model
import keras.backend as K
from keras.layers import BatchNormalization
import h5py
import numpy as np

from get_dataset_info import get_dataset_info


how_many_events=1
encoded_layer_no=14
modelpath = "models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_epoch96_supervised_up_down_epoch1.h5"

datatag1="xyzc"
datatag2="xyzc_flat_event"




homepath="/home/woody/capn/mppi013h/Km3-Autoencoder/"

dataset_dict1=get_dataset_info(datatag1)
dataset_dict2=get_dataset_info(datatag2)

test_file1=h5py.File(dataset_dict1["test_file"] , 'r')
test_file2=h5py.File(dataset_dict2["test_file"] , 'r')
model=load_model(homepath+modelpath)

data1=test_file1["x"][:how_many_events]
data2=test_file2["x"][:how_many_events]

def layer_output(input_data, layer_no, model, is_train_mode):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[layer_no].output])
    # output in test mode = 0
    layer_output = get_layer_output([input_data, is_train_mode])[0]
    return layer_output

def print_info(data1, model, encoded_layer_no):
    print("Layer:", model.layers[encoded_layer_no].name)
    encoded_train = layer_output(data1, encoded_layer_no, model, True)
    encoded_test = layer_output(data1, encoded_layer_no, model, False)
    for event_no in range(len(encoded_train)):
        print("Train:", np.mean(encoded_train[event_no]), "\tTest:",np.mean(encoded_test[event_no]))

def scan_layers(data, model, encoded_layer_no):
    for layer_no in range(encoded_layer_no):
        print_info(data,model,layer_no)
    
    
print(datatag1)

#print_info(data1,model,encoded_layer_no)
scan_layers(data1, model, encoded_layer_no)

for layer_no in range(encoded_layer_no):
    layer = model.layers[layer_no]
    if isinstance(layer, BatchNormalization):
        print(layer.name)
        layer._per_input_updates={}

scan_layers(data1, model, encoded_layer_no)


