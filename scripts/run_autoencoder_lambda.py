# -*- coding: utf-8 -*-

"""
FOR MODELS WITH LAMBDA LAYERS IN THE DECODER STAGE because they are bugged!
This script starts or resumes the training of an autoencoder or an encoder that is 
defined in model_definitions.
It also contatins the adress of the training files and the epoch
"""

from keras.models import load_model
from keras import optimizers
import numpy as np
import h5py
from run_cnn import *
from model_definitions import *
import os
import sys
import argparse


# start.py "vgg_1_xzt" 1 0 0 0 2 "up_down" True 0 11 18 50 1 
def parse_input():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('modeltag', type=str, help='an integer for the accumulator')
    parser.add_argument('runs', type=int)
    parser.add_argument("autoencoder_stage", type=int)
    parser.add_argument("autoencoder_epoch", type=int)
    parser.add_argument("encoder_epoch", type=int)
    parser.add_argument("class_type_bins", type=int)
    parser.add_argument("class_type_name", type=str)
    parser.add_argument("zero_center", type=bool)
    parser.add_argument("verbose", type=int)    
    parser.add_argument("n_bins", nargs=4, type=int)
    parser.add_argument("learning_rate", type=float)    
    
    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
modeltag = params["modeltag"]
runs=params["runs"]
autoencoder_stage=params["autoencoder_stage"]
autoencoder_epoch=params["autoencoder_epoch"]
encoder_epoch=params["encoder_epoch"]
class_type = (params["class_type_bins"], params["class_type_name"])
zero_center = params["zero_center"]
verbose=params["verbose"]
n_bins = params["n_bins"]
learning_rate = params["learning_rate"]


"""
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

#Learning rate, usually 0.00059
learning_rate = 0.00059
"""

#Naming scheme for models
"""
Autoencoder
"trained_" + modeltag + "_supervised_" + class_type[1] + "_epoch" + encoder_epoch + ".h5" 

Encoder+
"trained_" + modeltag + "_autoencoder_" + class_type[1] + "_epoch" + encoder_epoch + ".h5"

Encoder+ new
"trained_" + modeltag + "_autoencoder_" + autoencoder_epoch + "_supervised_" + class_type[1] + "_epoch" + encoder_epoch + ".h5" 
"""


def execute_training(modeltag, runs, autoencoder_stage, epoch, encoder_epoch, class_type, zero_center, verbose, n_bins, learning_rate):
    
    #Path to training and testing datafiles on HPC for xyz
    """
    data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/"
    train_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5"
    test_data = "test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5"
    zero_center_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5_zero_center_mean.npy"
    """
    
    #for xzt
    data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
    train_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
    test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
    
    
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
    
    #All models are now saved in their own folder   models/"modeltag"/
    model_folder = home_path + "models/" + modeltag + "/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    #Optimizer used in all the networks:
    lr = learning_rate # 0.01 default for SGD, 0.001 for Adam
    lr_decay = 0.05 # % decay for each epoch, e.g. if 0.1 -> lr_new = lr*(1-0.1)=0.9*lr
    #Default:
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam = optimizers.Adam(lr=lr,    beta_1=0.9, beta_2=0.999, epsilon=0.1,   decay=0.0)
    
    #fit_model and evaluate_model take lists of tuples, so that you can give many single files (here just one)
    train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
    test_tuple=[[test_file, h5_get_number_of_rows(test_file)]]
    
    
    #Check wheter a file with this name exists or not
    def check_for_file(proposed_filename):
        if(os.path.isfile(proposed_filename)):
            sys.exit(proposed_filename+ "exists already!")
    
    #Zero-Center with precalculated mean image
    xs_mean = np.load(zero_center_file) if zero_center is True else None
    
    
    #Setup network:
        
    #Autoencoder self-supervised training:
    if autoencoder_stage==0:
        modelname = modeltag + "_autoencoder"
        
        if epoch == 0:
            #Create a new autoencoder network
            
            model = setup_model(model_tag=modeltag, autoencoder_stage=0, modelpath_and_name=None)
            model.compile(optimizer=adam, loss='mse')
            
            #For a new autoencoder: Create header for test file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_file:
                metrics = str(model.metrics_names)
                if len(metrics)==1:
                    test_file.write('\n{0}\t{1}\t{2}'.format("Epoch", "LR", metrics[0]))
                elif len(metrics)==2:
                    test_file.write('\n{0}\t{1}\t{2}\t{3}'.format("Epoch", "LR", metrics[0], metrics[1]))
                else:
                    test_file.write('\n{0}\t{1}\t{2}'.format("Epoch", "LR", metrics))
            
            
        else:
            #Load an existing trained autoencoder network and train that
            
            #model = load_model(model_folder + "trained_" + modelname + '_epoch' + str(epoch) + '.h5')
            #in case of lambda layers: Load model structure and insert weights, because load model is bugged for lambda layers
            model=setup_model(model_tag=modeltag, autoencoder_stage=0, modelpath_and_name=None)
            model.load_weights(model_folder + "trained_" + modelname + '_epoch' + str(epoch) + '.h5', by_name=True)
            model.compile(optimizer=adam, loss='mse')
        
        
        #Execute training
        for current_epoch in range(epoch,epoch+runs):
            #Does the model we are about to save exist already?
            check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
            
            #Train network, write logfile, save network, evaluate network, save evaluation to file
            lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                                 batchsize=32, n_bins=n_bins, class_type=None, xs_mean=xs_mean, epoch=current_epoch,
                                 shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                                 save_path=model_folder, is_autoencoder=True, verbose=verbose)
    
            
    #Encoder supervised training:
    elif autoencoder_stage==1:
        #Load an existing autoencoder network, modify and train it supervised
        
        #name of the autoencoder model file that the encoder part is taken from:
        autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(epoch) + '.h5'
        
        #name of the supervised model:
        modelname = modeltag + "_autoencoder_epoch" + str(epoch) +  "_supervised_" + class_type[1]
        
        if encoder_epoch == 0:
            #Create a new encoder network:
            
            model = setup_model(model_tag=modeltag, autoencoder_stage=1, modelpath_and_name=autoencoder_model)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            
            #For a new encoder: Create header for test file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_file:
                metrics = str(model.metrics_names)
                if len(metrics)==1:
                    test_file.write('\n{0}\t{1}\t{2}'.format("Epoch", "LR", metrics[0]))
                elif len(metrics)==2:
                    test_file.write('\n{0}\t{1}\t{2}\t{3}'.format("Epoch", "LR", metrics[0], metrics[1]))
                else:
                    test_file.write('\n{0}\t{1}\t{2}'.format("Epoch", "LR", metrics))
            
        
        else:
            #Load an existing trained encoder network and train that
        
            model = load_model(model_folder + "trained_" + modelname + '_epoch' + str(encoder_epoch) + '.h5')
            #in case of lambda layers: Load model structure and insert weights, because load model is bugged for lambda layers
            #model=setup_model(model_tag=modeltag, autoencoder_stage=1, modelpath_and_name=None)
            #model.load_weights(model_folder + "trained_" + modelname + '_epoch' + str(epoch) + '.h5', by_name=True)
            #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        #Execute training
        for current_epoch in range(encoder_epoch,encoder_epoch+runs):
            #Does the model we are about to save exist already?
            check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
            
            #Train network, write logfile, save network, evaluate network
            lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                                 batchsize=32, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, epoch=current_epoch,
                                 shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                                 save_path=model_folder, is_autoencoder=False, verbose=verbose)
    
    
    
    #Encoder supervised training with completely unfrozen model:
    elif autoencoder_stage==2:
        #Load an existing autoencoder network, modify and train it supervised
        
        #name of the autoencoder model file that the encoder part is taken from:
        autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(epoch) + '.h5'
        
        #name of the supervised model:
        modelname = modeltag + "_supervised_" + class_type[1]
        
        if encoder_epoch == 0:
            #Create a new encoder network:
            
            model = setup_model(model_tag=modeltag, autoencoder_stage=2, modelpath_and_name=autoencoder_model)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            
            #For a new encoder: Create header for test file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_file:
                metrics = str(model.metrics_names)
                if len(metrics)==1:
                    test_file.write('\n{0}\t{1}\t{2}'.format("Epoch", "LR", metrics[0]))
                elif len(metrics)==2:
                    test_file.write('\n{0}\t{1}\t{2}\t{3}'.format("Epoch", "LR", metrics[0], metrics[1]))
                else:
                    test_file.write('\n{0}\t{1}\t{2}'.format("Epoch", "LR", metrics))
            
            
        
        else:
            #Load an existing trained encoder network and train that
        
            model = load_model(model_folder + "trained_" + modelname + '_epoch' + str(encoder_epoch) + '.h5')
            #in case of lambda layers: Load model structure and insert weights, because load model is bugged for lambda layers
            #model=setup_model(model_tag=modeltag, autoencoder_stage=2, modelpath_and_name=None)
            #model.load_weights(model_folder + "trained_" + modelname + '_epoch' + str(epoch) + '.h5', by_name=True)
            #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        #Execute training
        for current_epoch in range(encoder_epoch,encoder_epoch+runs):
            #Does the model we are about to save exist already?
            check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
            
            #Train network, write logfile, save network, evaluate network
            lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                                 batchsize=32, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, epoch=current_epoch,
                                 shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                                 save_path=model_folder, is_autoencoder=False, verbose=verbose)
    

execute_training(modeltag, runs, autoencoder_stage, autoencoder_epoch, encoder_epoch, class_type, zero_center, verbose, n_bins, learning_rate)
