# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input, Conv3D, UpSampling3D, Conv3DTranspose, AveragePooling3D
import numpy as np
import h5py
from run_cnn import *

def setup_conv_model_API():
    #Wie der autoencoder im sequential style, nur mit API
    
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
    return autoencoder

    
#Path to training and testing datafiles on HPC
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/"
train_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5"
test_data = "test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5"
train_file=data_path+train_data
test_file=data_path+test_data

#Path to the Km3_net-Autoencoder folder on HPC:
home_path="/home/woody/capn/mppi013h/Km3-Autoencoder"


#For debug testing on my laptop these are overwritten:
"""
home_path="../"
train_file="Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5"
#file=h5py.File(train_file, 'r')
#xyz_hists = np.array(file["x"]).reshape((3498,11,13,18,1))
"""
#fit_model and evaluate_model take lists of tuples, so that you can give many single files (here just one)
train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
test_tuple=[[test_file, h5_get_number_of_rows(test_file)]]

#Setup network:
autoencoder = setup_conv_model_API()
autoencoder.compile(optimizer='adam', loss='mse')

#Train network, write logfile, save network, evaluate network
train_and_test_model(model=autoencoder, modelname="autoencoder_test", train_files=train_tuple, test_files=test_tuple, batchsize=32, n_bins=(11,13,18,1), class_type=None, xs_mean=None, epoch=0,
                         shuffle=False, lr=None, lr_decay=None, tb_logger=False, swap_4d_channels=None, save_path=home_path)



#history = autoencoder.fit(xyz_hists[0:320], xyz_hists[0:320], epochs=1, batch_size=32)
#autoencoder.save('/home/woody/capn/mppi013h/Km3-Autoencoder/models/autoencoder_test.h5')

