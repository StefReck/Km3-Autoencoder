# -*- coding: utf-8 -*-

from run_autoencoder import *

#Tag for the model used; Identifies both autoencoder and encoder
modeltag="vgg_1_xzt"

#How many additinal epochs the network will be trained for by executing this script:
runs=1

#Type of training/network
# 0: autoencoder
# 1: encoder+ from autoencoder w/ frozen layers
# 2: encoder+ from scratch, completely unfrozen
autoencoder_stage=2

#Define starting epoch of autoencoder model
autoencoder_epoch=0

#If in encoder stage (1 or 2), encoder_epoch is used to identify a possibly
#existing supervised-trained encoder network
encoder_epoch=0
#Define what the supervised encoder network is trained for, and how many neurons are in the output
#This also defines the name of the saved model
class_type = (2, 'up_down')

#Wheter to use a precalculated zero-center image or not
zero_center = True

#Verbose bar during training?
#0: silent, 1:verbose, 2: one log line per epoch
verbose=0

# x  y  z  t
# 11 13 18 50
n_bins = (11,18,50,1)


#Path to training and testing datafiles on HPC for xyz
"""
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/"
train_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5"
test_data = "test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5_zero_center_mean.npy"
"""

#for xzt
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated"
train_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"


train_file=data_path+train_data
test_file=data_path+test_data
zero_center_file=data_path+zero_center_data



#Naming scheme for models
"""
Autoencoder
"trained_" + modeltag + "_supervised_" + class_type[1] + "_epoch" + encoder_epoch + ".h5" 

Encoder+
"trained_" + modeltag + "_autoencoder_" + class_type[1] + "_epoch" + encoder_epoch + ".h5"

Encoder+ new
"trained_" + modeltag + "_autoencoder_" + autoencoder_epoch + "_supervised_" + class_type[1] + "_epoch" + encoder_epoch + ".h5" 
"""


execute_training(modeltag, runs, autoencoder_stage, autoencoder_epoch, encoder_epoch, class_type, zero_center, n_bins)

