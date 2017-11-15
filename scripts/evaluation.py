# -*- coding: utf-8 -*-

"""
Evaluate model performance after training
"""

from util.evaluation_utilities import *
import numpy as np
from keras.models import load_model
import pickle

#Model info:
modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/vgg_1_xzt/"
#modelident="trained_vgg_1_xzt_autoencoder_epoch22_supervised_up_down_epoch10.h5"
modelident="trained_vgg_1_xzt_supervised_up_down_epoch6.h5"

#Info about model
n_bins = (11,18,50,1)
class_type = (2, 'up_down')
#Plot properties:
title_of_plot='Comparison'
#title_of_plot='Classification for up-dwon 3-100GeV unfrozen encoder+'
save_to = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/" + modelident[:-3] + "_Acc_plot" #plot and dumb will be saved


#Test data files:
#for xzt
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"




test_file=data_path+test_data
zero_center_file=data_path+zero_center_data
xs_mean = np.load(zero_center_file)
modelname = modelpath + modelident
model = load_model(modelname)


#load again
with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/trained_vgg_1_xzt_autoencoder_epoch22_supervised_up_down_epoch10arr_e_cor.txt", "rb") as dump_file:
    arr_energy_correct2 = pickle.load(dump_file)

arr_energy_correct = make_performance_array_energy_correct(model, test_file, n_bins, class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)

#save to file
#with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/" + modelident[:-3] + "arr_e_cor.txt", "wb") as dump_file:
#    pickle.dump(arr_energy_correct, dump_file)
    



bin_edges_centered, hist_1d_energy_accuracy_bins, hist_1d_energy_accuracy_bins2 = make_energy_to_accuracy_plot_comp(arr_energy_correct,arr_energy_correct2, title=title_of_plot, filepath=save_to)


#save to file
with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/" + modelident[:-3] + "arr_e_cor_newdumb.txt", "wb") as dump_file:
    pickle.dump([bin_edges_centered, hist_1d_energy_accuracy_bins, hist_1d_energy_accuracy_bins2], dump_file)
    

#make_energy_to_accuracy_plot_multiple_classes(arr_energy_correct, title=title_of_plot, filename=save_to)
#make_prob_hists(arr_energy_correct[:, ], modelname=modelident)
