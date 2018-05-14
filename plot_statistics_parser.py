# -*- coding: utf-8 -*-

import argparse


"""
Make a plot of multiple models, each identified with its test log file, with
lots of different options.
This is intended for plotting models with the same y axis data (loss OR acc).
"""

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training. Can also enter "saved" and a tag to restore saved plot properties.')
    parser.add_argument('models', type=str, nargs="+", help='name of _test.txt files to plot. (or "saved" and a tag)')

    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
test_files = params["models"]


#import matplotlib
import matplotlib.pyplot as plt

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y
from scripts.util.saved_setups_for_plot_statistics import get_props_for_plot_parser, get_plot_statistics_plot_size
#matplotlib.rcParams.update({'font.size': 14})




xlabel="Epoch"
title="Loss of autoencoders with a varying number of convolutional layers"
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
#How hte plotting window should look like
style="extended"
figsize, font_size = get_plot_statistics_plot_size(style)
#Save the plot, None to skip
save_as=None

#overwrite some of the above options from a saved setup
if test_files[0]=="saved":
    test_files, title, labels_override, save_as, legend_locations, colors, xticks, figsize, font_size = get_props_for_plot_parser(test_files[1])

#Read the data in
data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, 
                                                                        dump_to_file=dump_to_file)

#Create the plot
fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, 
                legend_locations, labels_override, colors, xticks, figsize=figsize, font_size=font_size)
#Save the plot
if save_as != None:
    plt.savefig(save_as)
    print("Saved plot as",save_as)

plt.show(fig)

