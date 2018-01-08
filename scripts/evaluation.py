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
Specify trained models, either AE or supervised, and calculate their loss or acc on test data.
This data is then binned and automatically dumped. Instead of recalculating, it is loaded automatically.
Can also plot it and save it to results/plots
"""

#Model info:
#list of modelidents to work on (has to be an array, so add , at the end if only one file)
modelidents = ("vgg_3/trained_vgg_3_autoencoder_epoch2_supervised_up_down_accdeg_epoch17.h5",
               "vgg_3/trained_vgg_3_autoencoder_epoch5_supervised_up_down_accdeg_epoch16.h5",
               "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_accdeg_epoch23.h5",
               "vgg_3/trained_vgg_3_autoencoder_epoch140_supervised_up_down_accdeg_epoch45.h5",)
#Which ones are autoencoders? Only relevant for generating new data
is_autoencoder_array = (0,0,0,0) 


#Plot properties: All in the array are plotted in one figure, with own label each
title_of_plot='MSE Loss of Autoencoder on xzt-Data'
label_array=["Autoencoder Epoch 2", "Epoch 5", "Epoch 10", "Epoch 140"]
plot_file_name = "vgg_3_autoencoder_accdeg_compare.pdf" #in the results/plots folder
#Type of plot which is generated for whole array (it should all be of the same type):
#loss, acc, None
plot_type = "loss"




#Info about model
n_bins = (11,18,50,1)
class_type = (2, 'up_down')

#Test data files:
#for xzt
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"

modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
#           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
for modelident in modelidents:
    modelnames.append(modelident.split("trained_")[1][:-3])

test_file=data_path+test_data
save_plot_as = plot_path + plot_file_name

zero_center_file=data_path+zero_center_data
xs_mean = np.load(zero_center_file)

#Accuracy as a function of energy binned to a histogramm. It is dumped automatically into the
#results/data folder, so that it has not to be generated again
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

#Loss of an AE as a function of energy, rest like above
def make_and_save_hist_data_autoencoder(modelpath, modelident, modelname, test_file, n_bins, class_type, xs_mean):
    model = load_model(modelpath + modelident)
    print("Making and saving energy_loss_array of autoencoder ", modelname)
    hist_data = make_autoencoder_energy_data(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)
    #save to file
    with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data

#open dumped histogramm data, that was generated from the above two functions
def open_hist_data(modelname):
    print("Opening existing hist_data of ", modelname)
    #load again
    with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "rb") as dump_file:
        hist_data = pickle.load(dump_file)
    return hist_data

def make_or_load_files(modelnames , modelidents=None, modelpath=None, test_file=None, n_bins=None, class_type=None, xs_mean=None, is_autoencoder_array=None):
    hist_data_array=[]
    for i,modelname in enumerate(modelnames):
        print("Working on ",modelname)
        name_of_file="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt"
        if os.path.isfile(name_of_file)==True:
            print("Loading existing hist_data file")
            hist_data_array.append(open_hist_data(modelname))
        else:
            if is_autoencoder_array[i]==0:
                hist_data = make_and_save_hist_data(modelpath, modelidents[i], modelname, test_file, n_bins, class_type, xs_mean)
            else:
                hist_data = make_and_save_hist_data_autoencoder(modelpath, modelidents[i], modelname, test_file, n_bins, class_type, xs_mean)
            hist_data_array.append(hist_data)
        print("Done.")
    return hist_data_array

#generate or load data automatically:
hist_data_array = make_or_load_files(modelnames , modelidents, modelpath, test_file, n_bins, class_type, xs_mean, is_autoencoder_array)

#generate data manually:
#for supervised networks:
#hist_data_array = make_and_save_hist_data(modelpath=modelpath, modelidents=modelidents, modelnames=modelnames, test_file=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean)
#for autoencoders:
#hist_data_array = make_and_save_hist_data_autoencoder(modelpath=modelpath, modelidents=modelidents, modelnames=modelnames, test_file=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean)

#Load data manually:
#hist_data_array = open_hist_data[modelnames]

#make plot of multiple data:
if plot_type == "acc":
    y_label_of_plot="Accuracy"
    make_energy_to_accuracy_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot) 
elif plot_type == "loss":
    y_label_of_plot="Loss"
    make_energy_to_loss_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot) 
elif plot_type == None:
    print("plot_type==None: Not generating plots")
else:
    print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")


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
    
