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


def get_props_for_plot(tag):
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/"
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
        
    else:
        print("Tag", tag, "unknown.")
        raise()
    test_files=[ae_model, prl_model]
    save_as=home+"results/plots/statistics/statistics_parallel_"+prl_model.split("/")[-1][:-4]+".pdf"
    return test_files, title, labels_override, save_as


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
    test_files, title, labels_override, save_as = get_props_for_plot(test_files[1])



#Which epochs from the parallel encoder history to take:
if params["channel"]==True:
    how_many_epochs_each_to_train = np.ones(100).astype(int)
else:
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