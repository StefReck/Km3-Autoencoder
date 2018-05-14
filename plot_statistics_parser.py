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

    parser.add_argument('-s', '--save', action="store_true", help='Save the plot to the path defined in the file.')
    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
test_files = params["models"]
save_it = params["save"]


#import matplotlib
import matplotlib.pyplot as plt

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y
from scripts.util.saved_setups_for_plot_statistics import get_props_for_plot_parser, get_plot_statistics_plot_size
#matplotlib.rcParams.update({'font.size': 14})

#Default Values:
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


if test_files[0]=="saved":
    #Load a saved setup, identified with a tag
    tag=test_files[1]
    if tag == "all":
        #make and save all the plots whose setup is saved, without displaying them.
        current_tag_number=0
        while True:
            try:
                test_files, title, labels_override, save_as, legend_locations, colors, xticks, figsize, font_size = get_props_for_plot_parser(tag)
                data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, 
                                                                            dump_to_file=dump_to_file)
                fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, 
                                legend_locations, labels_override, colors, xticks, figsize=figsize, font_size=font_size)
                fig.savefig(save_as)
                print("Saved plot as",save_as)
                plt.close(fig)
                current_tag_number+=1
            except NameError:
                print("Done. Generated a total of",current_tag_number,"plots.")
                break
    else:
        #overwrite some of the above options from a specific saved setup
        test_files, title, labels_override, save_as, legend_locations, colors, xticks, figsize, font_size = get_props_for_plot_parser(tag)

    #Read the data in
    data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, 
                                                                            dump_to_file=dump_to_file)
    #Create the plot
    fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, 
                    legend_locations, labels_override, colors, xticks, figsize=figsize, font_size=font_size)
    #Save the plot
    if save_as != None and save_it==True:
        plt.savefig(save_as)
        print("Saved plot as",save_as)
    else:
        print("Plot was not saved.")
    
    plt.show(fig)

else:
    #Load the models handed to the parser directly, without a tag
    
    #Read the data in
    data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, 
                                                                            dump_to_file=dump_to_file)
    #Create the plot
    fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, 
                    legend_locations, labels_override, colors, xticks, figsize=figsize, font_size=font_size)
    #Save the plot
    if save_as != None and save_it==True:
        plt.savefig(save_as)
        print("Saved plot as",save_as)
    else:
        print("Plot was not saved.")
    
    plt.show(fig)
