# -*- coding: utf-8 -*-
"""
Plot autoencoder and supervised parallel performance in one plot.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

from scripts.plotting.plot_statistics import make_data_from_files, get_default_labels, get_max_epoch

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
figsize = (9,6)
#Override default labels (names of the models); must be one for every test file, otherwise default
labels_override=[]
#legend location for the labels and the test/train box
legend_locations=(1, "upper left")
#Override xtick locations; None for automatic
xticks=None
# override line colors; must be one color for every test file, otherwise automatic
colors=[] # = automatic
#Name of file to save the numpy array with the plot data to; None will skip saving
dump_to_file=None



#Returns ( [[Test_epoch, Test_ydata, Train_epoch, Train_ydata], ...], ylabel_list) 
#for every test file
data_from_files, ylabel_list = make_data_from_files(test_files)
data_autoencoder = data_from_files[0]
data_parallel = data_from_files[1]

how_many_epochs_each_to_train = np.array([10,]*1+[2,]*5+[1,]*100)

take_these_prl_epochs=np.cumsum(how_many_epochs_each_to_train)
highest_ae_epoch = max(data_autoencoder[0])
take_these_prl_epochs=take_these_prl_epochs[take_these_prl_epochs<=highest_ae_epoch]


data_parallel = data_parallel[:,take_these_prl_epochs-1]


def make_plot_same_y(test_files, data_autoencoder, data_parallel, xlabel, ylabel_list, title, legend_locations, labels_override, colors, xticks, figsize): 
    fig, ax=plt.subplots(figsize=figsize)
    
    all_ylabels_equal = all(x == ylabel_list[0] for x in ylabel_list)
    if all_ylabels_equal == False:
        print("Warning: Not all ylabels are equal:", ylabel_list, ",\nchoosing ", ylabel_list[0] )

    
    label_array = get_default_labels(test_files)
    if len(labels_override) == len(label_array):
        label_array=labels_override
    else:
        print("Custom label array does not have the proper length (",len(label_array),"). Using default labels...")
    
    if len(colors) == len(label_array):
        color_override = True
    else:
        color_override = False
        print("color array does not have the rights size (", len(label_array), "), using default colors.")
    
    handles1=[]
    #plot the data in one plot
    #autoencoder
    if color_override==True:
        test_plot = ax.plot(data_autoencoder[0], data_autoencoder[1], marker="o", color=colors[0])
    else:
        test_plot = ax.plot(data_autoencoder[0], data_autoencoder[1], marker="o")
    #the train plot
    ax.plot(data_autoencoder[2], data_autoencoder[3], linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6)
    handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(), lw=3, label=label_array[0])
    handles1.append(handle_for_legend)
    
    #parallel
    ax2 = ax.twinx()
    if color_override==True:
        test_plot = ax2.plot(data_parallel[0], data_parallel[1], marker="o", color=colors[1])
    else:
        test_plot = ax2.plot(data_parallel[0], data_parallel[1], marker="o")
        
    handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(), lw=3, label=label_array[0])
    handles1.append(handle_for_legend)
    
    #lhandles, llabels = ax.get_legend_handles_labels()
    legend1 = plt.legend(handles=handles1, loc=legend_locations[0])
    
    test_line = mlines.Line2D([], [], color='grey', marker="o", label='Test')
    train_line = mlines.Line2D([], [], color='grey', linestyle="-", alpha=0.5, lw=2, label='Train')
    legend2 = plt.legend(handles=[test_line,train_line], loc=legend_locations[1])
    
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    max_epoch = get_max_epoch( [data_autoencoder, data_parallel] )
    plt.xlim((0.2,max_epoch))
    #plt.xticks( np.linspace(1,max_epoch,max_epoch) )
    
    if xticks is not None:
        plt.xticks( xticks )
    else:
        plt.xticks( np.arange(0, max_epoch+1,10) )
    ax.ylabel(ylabel_list[0])
    ax2.ylabel(ylabel_list[1])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True)
    return(fig)


