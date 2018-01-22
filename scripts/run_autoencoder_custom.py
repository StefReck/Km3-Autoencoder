# -*- coding: utf-8 -*-

from keras.models import load_model
from keras import optimizers
import numpy as np
from keras import backend as K

from util.run_cnn import *
from model_definitions import *

"""
Load a model and train it with another optimizer
"""


init_model="models/vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch60.h5"
save_as="models/vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch60_branch" #+_epochXX.h5

train_for_how_many_epochs=100

#The exponent of the epsilon to switch to: epsilon=10**epsilon_exp
epsilon_exp=-8


model = load_model(init_model)
#Set LR of loaded model to new lr
K.set_value(model.optimizer.epsilon, 10**epsilon_exp)


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
    

#fit_model and evaluate_model take lists of tuples, so that you can give many single files (here just one)
train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
test_tuple=[[test_file, h5_get_number_of_rows(test_file)]]


xs_mean = np.load(zero_center_file)

with open(test_logfile, 'w') as test_log_file:
    metrics = model.metrics_names #['loss', 'acc']
    test_log_file.write('{0}\tTest {1}\tTrain {2}\tTest {3}\tTrain {4}\tTime\tLR'.format("Epoch", metrics[0], metrics[0],metrics[1],metrics[1]))

    
model.summary()
print("Model: ", init_model)
print("Current State of optimizer: \n", model.optimizer.get_config())
print("Train files:", train_tuple)
print("Test files:", test_tuple)

#Execute Training:
for current_epoch in range(running_epoch,running_epoch+runs):
    #Does the model we are about to save exist already?
    check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
    
    #Train network, write logfile, save network, evaluate network, save evaluation to file
    lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                         batchsize=32, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, epoch=current_epoch,
                         shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                         save_path=model_folder, is_autoencoder=is_autoencoder, verbose=verbose)    
            
        