# -*- coding: utf-8 -*-
"""
C
"""
from keras.models import load_model
import keras.backend as K
import h5py

from get_dataset_info import get_dataset_info


how_many_events=32
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

encoded_train = layer_output(data1, encoded_layer_no, model, True)
encoded_test = layer_output(data1, encoded_layer_no, model, False)

print("Train:", encoded_train)
print("Test:", encoded_test)

