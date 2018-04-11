# -*- coding: utf-8 -*-
"""
Plot autoencoder and supervised parallel performance in one plot.
"""

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training. Can also enter "saved" and a tag to restore saved plot properties.')
    parser.add_argument('autoencoder_model', type=str, help='name of _test.txt files to plot (or "saved")')
    parser.add_argument('parallel_model', type=str, help='name of _test.txt files to plot. (or a tag)')
    
    parser.add_argument('-c', '--channel', action="store_true", help='Use the channel parallel schedule (1 enc epoch per AE epoch)')
    args = parser.parse_args()
    params = vars(args)
    return params
params = parse_input()

import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y_parallel, get_last_prl_epochs


test_files = [ params["autoencoder_model"], params["parallel_model"] ]

#Epoch schedule that was used during training:
if params["channel"]==True:
    #used for channel AE
    epoch_schedule="1-1-1"
else:
    #used for everything else
    epoch_schedule="10-2-1"
    

def get_props_for_plot(tag):
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/"
    epoch_schedule="10-2-1"
    if tag=="msep":
        title = "Parallel training with MSEp autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]
    elif tag=="msepsq":
        title = r"Parallel training with MSEp$^2$ autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]    
    elif tag=="msep2":
        title = "Parallel training with MSEp autoencoder loss (low lr)"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        
    elif tag=="channel_3n":
        title = "Parallel training with channel autoencoder (3 neurons)"
        ae_model =  home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        epoch_schedule="1-1-1"
    elif tag=="channel_5n":
        title = "Parallel training with channel autoencoder (5 neurons)"
        ae_model =  home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        labels_override = ["Autoencoder", "Encoder"]  
        epoch_schedule="1-1-1"
    elif tag=="channel_10n":
        title = "Parallel training with channel autoencoder (10 neurons)"
        ae_model =  home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        labels_override = ["Autoencoder", "Encoder"]  
        epoch_schedule="1-1-1"
        
    elif tag=="vgg_5_200":
        title = "Parallel training with model '200'"
        ae_model =  home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]  
    elif tag=="vgg_5_200_dense":
        title = "Parallel training with model '200 dense'"
        ae_model =  home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]       
        
    elif tag=="vgg_5_200_deep":
        title = "Parallel training with model '200 deep'"
        ae_model =  home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_200_large":
        title = "Parallel training with model '200 large'"
        ae_model =  home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    
        
    elif tag=="vgg_5_64":
        title = "Parallel training with model '64'"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_32":
        title = "Parallel training with model '32' and $\epsilon=10^{-8}$"
        ae_model =  home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_32-eps01":
        title = "Parallel training with model '32'"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        
    else:
        print("Tag", tag, "unknown.")
        raise()
    test_files=[ae_model, prl_model]
    save_as=home+"results/plots/statistics/statistics_parallel_"+prl_model.split("/")[-1][:-4]+".pdf"
    return test_files, title, labels_override, save_as, epoch_schedule


xlabel="Epoch"
title="Parallel Training"
figsize = (13,8)
#Override default labels (names of the models); must be one for every test file, otherwise default
labels_override=[]
#legend location for the labels and the test/train box
legend_locations=(1, "upper left")
#Override xtick locations; None for automatic
xticks=None
# override line colors; must be one color for every test file, otherwise automatic
colors=["blue", "orange"] # = automatic
#Name of file in the results/dumped_statistics folder to save the numpy array 
#with the plot data to; None will skip saving
dump_to_file=None



#save plot as
save_as=None
if test_files[0]=="saved":
    test_files, title, labels_override, save_as, epoch_schedule = get_props_for_plot(test_files[1])



#Which epochs from the parallel encoder history to take:
if epoch_schedule=="1-1-1":
    how_many_epochs_each_to_train = np.ones(100).astype(int)
elif epoch_schedule=="10-2-1":
    how_many_epochs_each_to_train = np.array([10,]*1+[2,]*5+[1,]*200)
print("Using parallel schedule", how_many_epochs_each_to_train[:12,], "...")



#Returns ( [[Test_epoch, Test_ydata, Train_epoch, Train_ydata], ...], ylabel_list, default_label_array) 
#for every test file
data_from_files, ylabel_list, default_label_array = make_data_from_files(test_files, dump_to_file)

data_autoencoder = data_from_files[0]
data_parallel = data_from_files[1]

data_parallel_test, data_parallel_train = get_last_prl_epochs(data_autoencoder, data_parallel, how_many_epochs_each_to_train)


fig = make_plot_same_y_parallel(data_autoencoder, data_parallel_train, data_parallel_test, default_label_array, xlabel, ylabel_list, 
                 title, legend_locations, labels_override, colors, xticks, figsize)

if save_as != None:
    plt.savefig(save_as)
    print("Saved plot as",save_as)
    
plt.show(fig)