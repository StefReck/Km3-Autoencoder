# -*- coding: utf-8 -*-

"""
This script starts or resumes the training of an autoencoder or an encoder that is 
defined in model_definitions.
It also contatins the adress of the training files and the epoch
"""

from keras.models import load_model
import numpy as np
import h5py
from run_cnn import *
from model_definitions import *
import os.path
import sys


#How many additinal epochs the network will be trained for by executing this script:
runs=2

#Wheter this is self-supervised autoencoder training or supervised encoder training
autoencoder_stage=True

#If autoencoder_stage==False, encoder_epoch is used to identify a possibly
#existing supervised-trained encoder-stage network
encoder_epoch=0

#Define starting epoch and name of autoencoder model
epoch=1

#Tag for the model used; Identifies both autoencoder and encoder
modeltag="vgg_0"






#Path to training and testing datafiles on HPC
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/"
train_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5"
test_data = "test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5_zero_center_mean.npy"
train_file=data_path+train_data
test_file=data_path+test_data
zero_center_file=data_path+zero_center_data


#Path to my Km3_net-Autoencoder folder on HPC:
home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"


#For debug testing on my laptop these are overwritten:
"""
home_path="../"
train_file="Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5"
test_file="Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5"
#file=h5py.File(train_file, 'r')
#xyz_hists = np.array(file["x"]).reshape((3498,11,13,18,1))
"""


#fit_model and evaluate_model take lists of tuples, so that you can give many single files (here just one)
train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
test_tuple=[[test_file, h5_get_number_of_rows(test_file)]]


#Check wheter a file with this name exists or not
def check_for_file(proposed_filename):
    if(os.path.isfile(proposed_filename)):
        sys.exit(proposed_filename+ "exists already!")


#Setup network:
    
#Autoencoder self-supervised training:
if autoencoder_stage==True:
    modelname = modeltag + "_autoencoder"
    
    if epoch == 0:
        #Create a new autoencoder network
        
        model = setup_vgg_like(make_autoencoder=True)
        #Default: keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer='adam', loss='mse')
        
    else:
        #Load an existing trained autoencoder network and train that
        
        model = load_model(home_path+"models/trained_" + modelname + '_epoch' + str(epoch) + '.h5')
    
    
    #Execute training
    for current_epoch in range(epoch,epoch+runs):
        #Does the model we are about to save exist already?
        check_for_file(home_path+"models/trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
        
        #Train network, write logfile, save network, evaluate network, save evaluation to file
        train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                             batchsize=32, n_bins=(11,13,18,1), class_type=None, xs_mean=None, epoch=current_epoch,
                             shuffle=False, lr=None, lr_decay=None, tb_logger=False, swap_4d_channels=None,
                             save_path=home_path)

        
#Encoder supervised training:
elif autoencoder_stage==False:
    #Load an existing autoencoder network, modify and train it supervised
    modelname = modeltag + "_supervised"
    
    if epoch == 0:
        #Create a new enocder network:
        
        model = setup_vgg_like(make_autoencoder=False, 
                               modelpath_and_name=home_path+"models/trained_" + modelname + '_epoch' + str(epoch) + '.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    else:
        #Load an existing trained encoder network and train that
    
        model = load_model(home_path+"models/trained_" + modelname + '_epoch' + str(epoch) + '.h5')
    
    
    #Execute training
    for current_epoch in range(epoch,epoch+runs):
        #Does the model we are about to save exist already?
        check_for_file(home_path+"models/trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
        
        #Train network, write logfile, save network, evaluate network
        train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                             batchsize=32, n_bins=(11,13,18,1), class_type=None, xs_mean=None, epoch=current_epoch,
                             shuffle=False, lr=None, lr_decay=None, tb_logger=False, swap_4d_channels=None,
                             save_path=home_path)





