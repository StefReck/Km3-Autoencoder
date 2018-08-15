# -*- coding: utf-8 -*-

"""
Plot the training history of a model.

This will look up the logfiles of the model, and plot train and test loss over
the epoch.
Designed for matplotlib v2.0.1.
"""

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Make overview plots of model training. Can also enter "saved" and a tag to restore saved plot properties.')
    parser.add_argument('models', type=str, nargs="+", help='name of _test.txt files to plot. (or "saved" and a tag)')

    parser.add_argument('-s', '--save', action="store_true", help='Save the plot to the path defined in the file.')
    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
test_files = params["models"]
save_it = params["save"]

import matplotlib.pyplot as plt

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y
from scripts.util.saved_setups_for_plot_statistics import get_props_for_plot_parser

#Properties of the plot
#These will be overwritten if using a saved setup.
xlabel="Epoch"
title=""
#Override default labels (names of the models); must be one for every test file, otherwise default
labels_override=["model",]
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
#Save the plot, None to skip
save_as=None
#Ranges for the plot
xrange="auto"
#Average over this many bins in the train data (to reduce jitter)
average_train_data_bins=1


def make_parser_plot(test_files, title, labels_override, save_as, 
                     legend_locations, colors, xticks, style,
                     dump_to_file, xlabel, save_it, show_it, xrange, 
                     average_train_data_bins=1):
    """ Generate the plot. Can also save or display it. """
    
    #Read the data in, auto means take acc when available, otherwise loss
    data_for_plots, ylabel_list, default_label_array = make_data_from_files(
            test_files, which_ydata="auto", dump_to_file=dump_to_file)
    #Create the plot
    fig = make_plot_same_y(
            data_for_plots, default_label_array, xlabel, ylabel_list, title, 
            legend_locations, labels_override, colors, xticks, style=style, 
            xrange=xrange, average_train_data_bins=average_train_data_bins)
    
    #Save the plot
    if save_as != None and save_it==True:
        plt.savefig(save_as)
        print("Saved plot as",save_as)
    else:
        print("Plot was not saved.")
    if show_it: plt.show(fig)

def make_plot_for_saved_setup(tag):
    """ Use a saved tag to load all info, then plot it. """
    
    if tag == "all":
        #make and save all the plots whose setup is saved, without displaying them.
        current_tag_number=0
        while True:
            try:
                (test_files, title, labels_override, save_as, legend_locations,
                 colors, xticks, style, xrange, average_train_data_bins
                 ) = get_props_for_plot_parser(current_tag_number)
                make_parser_plot(test_files, title, labels_override, save_as, 
                     legend_locations, colors, xticks, style,
                     dump_to_file, xlabel, save_it=True, show_it=False,
                     xrange=xrange, average_train_data_bins=average_train_data_bins)
                current_tag_number+=1
            except NameError:
                print("Done. Generated a total of",current_tag_number,"plots.")
                break
    else:
        #Only plot a sepcific saved setup and show it
        (test_files, title, labels_override, save_as, legend_locations, 
         colors, xticks, style, xrange, average_train_data_bins) = get_props_for_plot_parser(tag)

        make_parser_plot(test_files, title, labels_override, save_as, 
                     legend_locations, colors, xticks, style,
                     dump_to_file, xlabel, save_it, show_it=True, 
                     xrange=xrange, average_train_data_bins=average_train_data_bins)


if test_files[0]=="saved":
    #Load a saved setup, identified with a tag
    tag=test_files[1]
    make_plot_for_saved_setup(tag)
else:
    #Load the models handed to the parser directly, without a tag
    make_parser_plot(test_files, title, labels_override, save_as, 
                     legend_locations, colors, xticks, style,
                     dump_to_file, xlabel, save_it, show_it=True, xrange=xrange,
                     average_train_data_bins=average_train_data_bins)
