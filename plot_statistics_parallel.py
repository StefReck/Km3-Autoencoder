# -*- coding: utf-8 -*-
"""
Plot autoencoder and supervised parallel performance in one plot.
"""
#TODO display train loss of parallel where possible (when schedule is at 1 AE E/1 par E)
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

from scripts.plotting.plot_statistics import make_data_from_files, get_default_labels, get_max_epoch, make_plot_same_y_parallel

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training.')
    parser.add_argument('autoencoder_model', type=str, help='name of _test.txt files to plot.')
    parser.add_argument('parallel_model', type=str, help='name of _test.txt files to plot.')
    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
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
#Name of file to save the numpy array with the plot data to; None will skip saving
dump_to_file=None



#Returns ( [[Test_epoch, Test_ydata, Train_epoch, Train_ydata], ...], ylabel_list) 
#for every test file
data_from_files, ylabel_list = make_data_from_files(test_files)
data_autoencoder = data_from_files[0]
data_parallel = data_from_files[1]
highest_prl_epoch = max(data_parallel[0])

#Which epochs from the parallel encoder history to take:
how_many_epochs_each_to_train = np.array([10,]*1+[2,]*5+[1,]*200)
take_these_prl_epochs=np.cumsum(how_many_epochs_each_to_train)
take_these_prl_epochs=take_these_prl_epochs[take_these_prl_epochs<=highest_prl_epoch]

data_parallel_test = np.array(data_parallel[0:2])[:,take_these_prl_epochs-1]


#train: Only take epochs that were trained for one Epoch on an AE Epoch
is_1=np.where(how_many_epochs_each_to_train==1)[0][0]-1
take_these_train_epochs = take_these_prl_epochs[is_1:]
#shift epochs, so that it will be plotted over the AE epoch and not the spvsd epoch
shift_epochs_by = take_these_prl_epochs[is_1] - (is_1+1) 

#data_parallel_train=[train_epoch, train_ydata]
data_parallel_train=[[],[]]
for epoch in take_these_train_epochs:
    take_these = np.logical_and(data_parallel[2]>=epoch, data_parallel[2]<epoch+1)
    data_parallel_train[0].extend( (np.array(data_parallel[2])-shift_epochs_by)[take_these])
    data_parallel_train[1].extend( (np.array(data_parallel[3])-shift_epochs_by)[take_these])
    




fig = make_plot_same_y_parallel(test_files, data_autoencoder, data_parallel_train, data_parallel_test, xlabel, ylabel_list, 
                 title, legend_locations, labels_override, colors, xticks, figsize)
plt.show(fig)
