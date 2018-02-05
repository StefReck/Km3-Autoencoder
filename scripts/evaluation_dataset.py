# -*- coding: utf-8 -*-

"""
Evaluate model performance after training. 
This is for comparison of supervised accuracy on different datasets.
"""

import numpy as np
from keras.models import load_model
import pickle
import os

from util.evaluation_utilities import *
from util.run_cnn import load_zero_center_data, h5_get_number_of_rows
from get_dataset_info import get_dataset_info


"""
Specify trained models and calculate their loss or acc on test data.
This data is then binned and automatically dumped. Instead of recalculating, it is loaded automatically.
"""



#Model info:
#list of modelidents to work on (has to be an array, so add , at the end if only one file)
"""
modelidents = ("vgg_3-broken1/trained_vgg_3-broken1_supervised_up_down_epoch6.h5",
               "vgg_3-broken1/trained_vgg_3-broken1_supervised_up_down_epoch6.h5",
               "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5")
"""

modelidents = ("vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_BN_unlocked_epoch1.h5",
               "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_BN_unlocked_epoch2.h5",
               "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_BN_unlocked_epoch3.h5",
               "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_epoch14.h5")

#Which dataset each to use
dataset_array = ("xzt", "xzt", "xzt")


#Plot properties: All in the array are plotted in one figure, with own label each
title_of_plot='Autoencoder-encoder network performance with unlocked batch norm'
label_array=["Epoch 1", "Epoch 2", "Epoch 3", "Frozen BN"]
#Overwrite default color palette. Leave empty for auto
color_array=["blue", "orange", "green", "grey"]

plot_file_name = "vgg_3_broken_encoder_acc_BN_unlocked.pdf" #in the results/plots folder
#Type of plot which is generated for whole array (it should all be of the same type):
#loss, acc, None
plot_type = "acc"
#y limits of plot:
y_lims=(0.4,1.0)

#Info about model
class_type = (2, 'up_down')


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"

modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
#           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
for modelident in modelidents:
    modelnames.append(modelident.split("trained_")[1][:-3])
    
save_plot_as = plot_path + plot_file_name
    

#Accuracy as a function of energy binned to a histogramm. It is dumped automatically into the
#results/data folder, so that it has not to be generated again
def make_and_save_hist_data(modelpath, dataset, modelident, class_type, name_of_file):
    model = load_model(modelpath + modelident)
    
    dataset_info_dict = get_dataset_info(dataset)
    #home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"]
    
    train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=32, n_bins=n_bins, n_gpu=1)

    
    print("Making energy_coorect_array of ", modelident)
    arr_energy_correct = make_performance_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, broken_simulations_mode=broken_simulations_mode, swap_4d_channels=None, samples=None)
    #hist_data = [bin_edges_centered, hist_1d_energy_accuracy_bins]:
    hist_data = make_energy_to_accuracy_data(arr_energy_correct, plot_range=(3,100))
    #save to file
    print("Saving hist_data as", name_of_file)
    with open(name_of_file, "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data


"""
#Loss of an AE as a function of energy, rest like above
def make_and_save_hist_data_autoencoder(modelpath, modelident, modelname, test_file, n_bins, class_type, xs_mean):
    model = load_model(modelpath + modelident)
    print("Making and saving energy_loss_array of autoencoder ", modelname)
    hist_data = make_autoencoder_energy_data(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)
    #save to file
    with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data
"""


#open dumped histogramm data, that was generated from the above two functions
def open_hist_data(name_of_file):
    print("Opening existing hist_data file", name_of_file)
    #load again
    with open(name_of_file, "rb") as dump_file:
        hist_data = pickle.load(dump_file)
    return hist_data

def make_or_load_files(modelnames, dataset_array, modelidents=None, modelpath=None, class_type=None):
    hist_data_array=[]
    for i,modelname in enumerate(modelnames):
        dataset=dataset_array[i]
        
        print("Working on ",modelname,"using dataset", dataset)
        name_of_file="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_" + dataset + "_hist_data.txt"
        if os.path.isfile(name_of_file)==True:
            hist_data_array.append(open_hist_data(name_of_file))
        else:
            hist_data = make_and_save_hist_data(modelpath, dataset, modelidents[i], class_type, name_of_file)
            hist_data_array.append(hist_data)
        print("Done.")
    return hist_data_array



#generate or load data automatically:
hist_data_array = make_or_load_files(modelnames, dataset_array, modelidents, modelpath, class_type)

#make plot of multiple data:
if plot_type == "acc":
    y_label_of_plot="Accuracy"
    make_energy_to_accuracy_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array) 
elif plot_type == "loss":
    y_label_of_plot="Loss"
    make_energy_to_loss_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, color_array=color_array) 
elif plot_type == None:
    print("plot_type==None: Not generating plots")
else:
    print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")

print("Plot saved to", save_plot_as)
