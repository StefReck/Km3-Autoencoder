# -*- coding: utf-8 -*-
"""
Plot autoencoder and supervised parallel performance in one plot.
"""

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training.')
    parser.add_argument('autoencoder_model', type=str, help='name of _test.txt files to plot.')
    parser.add_argument('parallel_model', type=str, help='name of _test.txt files to plot.')
    
    parser.add_argument('-c', '--channel', action="store_true", help='Use the channel parallel schedule (1 enc epoch per AE epoch)')
    args = parser.parse_args()
    params = vars(args)
    return params
params = parse_input()

import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y_parallel, get_last_prl_epochs


test_files = [ params["autoencoder_model"], params["parallel_model"] ]


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


#Which epochs from the parallel encoder history to take:
if params["channel"]==True:
    how_many_epochs_each_to_train = np.ones(100)
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
plt.show(fig)
