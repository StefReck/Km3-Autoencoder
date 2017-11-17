# -*- coding: utf-8 -*-

"""
Evaluate model performance after training
"""

from util.evaluation_utilities import *
import numpy as np
from keras.models import load_model
import pickle

"""
Take all models in the modelident array and make binned accuracy histogramm data and save.
Autoencoder loss has to be done seperately.
Can also open saved ones to plot all of those histgramms to one plot.
"""

#Model info:
modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
#list of modelidents to work
modelidents = ("vgg_1_xzt/trained_vgg_1_xzt_autoencoder_epoch43.h5",
               "vgg_1_xzt/trained_vgg_1_xzt_autoencoder_epoch22.h5")
label_array=["Epoch 43","Epoch 22"]

#Info about model
n_bins = (11,18,50,1)
class_type = (2, 'up_down')
#Plot properties:
title_of_plot='Loss of autoencoder (xzt Data)'


modelnames=[] # a tuple of eg "vgg_1_xzt_supervised_up_down_epoch6"
for modelident in modelidents:
    modelnames.append(modelident.split("trained_")[1][:-3])

#title_of_plot='Classification for up-dwon 3-100GeV unfrozen encoder+'
save_plot_as = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/" + modelnames[0] + "_" + modelnames[1] + "_Acc_Comp.pdf"


#Test data files:
#for xzt
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"

test_file=data_path+test_data
zero_center_file=data_path+zero_center_data
xs_mean = np.load(zero_center_file)



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

#for supervised networks:
hist_data_array = make_and_save_hist_data(modelpath=modelpath, modelidents=modelidents, modelnames=modelnames, test_file=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean)
#for autoencoders:
#hist_data_array = make_and_save_hist_data_autoencoder(modelpath=modelpath, modelidents=modelidents, modelnames=modelnames, test_file=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean)
#for saved hist data:
#hist_data_array = open_hist_data[modelnames]
# make plot of multiple data:
make_energy_to_accuracy_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, ylabel="Loss (mse)") 



