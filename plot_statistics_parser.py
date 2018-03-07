# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import argparse

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y
matplotlib.rcParams.update({'font.size': 14})

"""
Make a plot of multiple models, each identified with its test log file, with
lots of different options.
This is intended for plotting models with the same y axis data (loss OR acc).
"""

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training.')
    parser.add_argument('models', type=str, nargs="+", help='name of _test.txt files to plot.')

    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
test_files = params["models"]

xlabel="Epoch"
title="Loss of autoencoders with a varying number of convolutional layers"
figsize = (13,8)
#Override default labels (names of the models); must be one for every test file, otherwise default
labels_override=["12 layers CW", "12 layers","14 layers", "16 layers", "20 layers"]
#legend location for the labels and the test/train box
legend_locations=(1, "upper left")
#Override xtick locations; None for automatic
xticks=None
# override line colors; must be one color for every test file, otherwise automatic
colors=[] # = automatic
#Name of file to save the numpy array with the plot data to; None will skip saving
dump_to_file=None







data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, dump_to_file=dump_to_file)
fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, 
                legend_locations, labels_override, colors, xticks, figsize=figsize)
plt.show(fig)

