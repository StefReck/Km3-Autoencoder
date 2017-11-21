# -*- coding: utf-8 -*-

"""
Evaluate model performance after training
"""

import numpy as np
from keras.models import load_model
import pickle
import os

from util.evaluation_utilities import *

"""
Take all models in the modelident array and make binned accuracy histogramm data and save.
Autoencoder loss has to be done seperately.
Can also open saved ones to plot all of those histgramms to one plot.
"""

#Model info:
modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
#list of modelidents to work (has to be an array, so add , at the end if only one file)
modelidents = ("vgg_1_xzt/trained_vgg_1_xzt_autoencoder_epoch40.h5",)
is_autoencoder_array = (1,) #Which ones are autoencoders? Only relevant for generating new data
label_array=["Autoencoder Epoch 43",]

y_label_of_plot="Loss"

#Plot properties:
title_of_plot='MSE Loss of Autoencoder on xzt-Data'
save_plot_as = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/vgg_1_xzt_autoencoder_epoch_43_loss.pdf"

#Info about model
n_bins = (11,18,50,1)
class_type = (2, 'up_down')




modelnames=[] # a tuple of eg "vgg_1_xzt_supervised_up_down_epoch6" (created from trained_vgg_1_xzt_supervised_up_down_epoch6.h5)
for modelident in modelidents:
    modelnames.append(modelident.split("trained_")[1][:-3])




#Test data files:
#for xzt
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"

test_file=data_path+test_data
zero_center_file=data_path+zero_center_data
xs_mean = np.load(zero_center_file)



def make_and_save_hist_data(modelpath, modelident, modelname, test_file, n_bins, class_type, xs_mean):
    model = load_model(modelpath + modelident)
    print("Making energy_coorect_array of ", modelname)
    arr_energy_correct = make_performance_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)
    #hist_data = [bin_edges_centered, hist_1d_energy_accuracy_bins]:
    hist_data = make_energy_to_accuracy_data(arr_energy_correct, plot_range=(3,100))
    #save to file
    print("Saving hist_data of ", modelname)
    with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data

def make_and_save_hist_data_autoencoder(modelpath, modelident, modelname, test_file, n_bins, class_type, xs_mean):
    model = load_model(modelpath + modelident)
    print("Making and saving energy_loss_array of autoencoder ", modelname)
    hist_data = make_autoencoder_energy_data(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)
    #save to file
    with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data

def open_hist_data(modelname):
    print("Opening existing hist_data of ", modelname)
    #load again
    with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "rb") as dump_file:
        hist_data = pickle.load(dump_file)
    return hist_data

def make_or_load_files(modelnames , modelidents=None, modelpath=None, test_file=None, n_bins=None, class_type=None, xs_mean=None, is_autoencoder_array=None):
    hist_data_array=[]
    for i,modelname in enumerate(modelnames):
        name_of_file="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt"
        if os.path.isfile(name_of_file)==True:
            hist_data_array.append(open_hist_data(modelname))
        else:
            if is_autoencoder_array[i]==0:
                hist_data = make_and_save_hist_data(modelpath, modelidents[i], modelname, test_file, n_bins, class_type, xs_mean)
            else:
                hist_data = make_and_save_hist_data_autoencoder(modelpath, modelidents[i], modelname, test_file, n_bins, class_type, xs_mean)
            hist_data_array.append(hist_data)
    return hist_data_array

hist_data_array = make_or_load_files(modelnames , modelidents, modelpath, test_file, n_bins, class_type, xs_mean, is_autoencoder_array)


#for supervised networks:
#hist_data_array = make_and_save_hist_data(modelpath=modelpath, modelidents=modelidents, modelnames=modelnames, test_file=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean)
#for autoencoders:
#hist_data_array = make_and_save_hist_data_autoencoder(modelpath=modelpath, modelidents=modelidents, modelnames=modelnames, test_file=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean)
#for saved hist data:
#hist_data_array = open_hist_data[modelnames]
# make plot of multiple data:
#make_energy_to_accuracy_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot) 
#make_energy_to_loss_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot) 



#For an array of data:
"""
def make_and_save_hist_data(modelpath, modelidents, modelnames, test_file, n_bins, class_type, xs_mean):
    hist_data_array=[]
    for i, modelident in enumerate(modelidents):
        model = load_model(modelpath + modelident)
        print("Making energy_coorect_array of ", modelnames[i])
        arr_energy_correct = make_performance_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)
        #hist_data = [bin_edges_centered, hist_1d_energy_accuracy_bins]:
        hist_data = make_energy_to_accuracy_data(arr_energy_correct, plot_range=(3,100))
        #save to file
        print("Saving hist_data of ", modelnames[i])
        with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelnames[i] + "_hist_data.txt", "wb") as dump_file:
            pickle.dump(hist_data, dump_file)
        hist_data_array.append(hist_data)
    return hist_data_array

def make_and_save_hist_data_autoencoder(modelpath, modelidents, modelnames, test_file, n_bins, class_type, xs_mean):
    hist_data_array=[]
    for i, modelident in enumerate(modelidents):
        model = load_model(modelpath + modelident)
        print("Making and saving energy_loss_array of autoencoder ", modelnames[i])
        hist_data = make_autoencoder_energy_data(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)
        #save to file
        with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelnames[i] + "_hist_data.txt", "wb") as dump_file:
            pickle.dump(hist_data, dump_file)
        hist_data_array.append(hist_data)
    return hist_data_array


def open_hist_data(modelnames):
    hist_data_array=[]
    for modelname in modelnames:
        print("Opening existing hist_data of ", modelname)
        #load again
        with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "rb") as dump_file:
            hist_data = pickle.load(dump_file)
        hist_data_array.append(hist_data)
    return hist_data_array
"""
    
