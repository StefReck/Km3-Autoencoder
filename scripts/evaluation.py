# -*- coding: utf-8 -*-

"""
        Evalutaion for up-down classification models
        
Evaluate model performance after training for up-down or AUtoencoder networks.
Will save and display a binned histogram plot of acc or loss vs energy.

Specify trained models, either AE or supervised up-down, and calculate their loss or acc
on test data.
This is then binned and automatically dumped. Instead of recalculating, 
it is loaded automatically.
Can also plot it and save it to results/plots
"""
import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Take a model that predicts energy of events and do the evaluation for that, either in the form of a 2d histogramm (mc energy vs reco energy), or as a 1d histogram (mc_energy vs mean absolute error).')
    parser.add_argument('tag', type=str, help='Name of an identifier for a saved setup. (see this file for identifiers)')

    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()

import matplotlib.pyplot as plt
import numpy as np

from util.evaluation_utilities import make_or_load_files, make_binned_data_plot
from util.saved_setups_for_plot_statistics import get_path_best_epoch


tag=params["tag"]

def get_saved_plots_info(tag):
    class_type = (2, 'up_down')
    #Type of plot which is generated for whole array (it should all be of the same type):
    #loss, acc, None
    plot_type = "acc"
    bins=32
    y_lims=(0.75,1) #for acc only
    title_of_plot=""
    #default plot size: two in one line, see make_binned_data_plot
    
    modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/updown_evaluation/"
    
    
    full_path=False
    if tag=="dpg_plot":
        #Model info:
        #list of modelidents to work on (has to be an array, so add , at the end
        #if only one file)
        modelidents = (get_path_best_epoch("vgg_5_200", full_path),
                       get_path_best_epoch("vgg_3", full_path),
                       get_path_best_epoch("vgg_3_unf", full_path))
        #Dataset to evaluate on
        dataset_array = ["xzt",] * len(modelidents)
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Up-down Performance comparison'
        label_array=["Best from autoencoder 200", "Best from autoencoder 2000", "Best supervised"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "dpg_vgg3_vgg5_200_spvsd_comp.pdf" 
    
    elif tag=="compare_2000_sup":
        #morefilter and picture (evtl channel)
        modelidents = (get_path_best_epoch("vgg_3_unf", full_path),
                       get_path_best_epoch("vgg_3", full_path),)
        dataset_array = ["xzt",] * len(modelidents)
        #title_of_plot='Accuracy of encoders with bottleneck 600'
        label_array=["Supervised approach", "From autoencoder $\epsilon=10^{-1}$ epoch 10"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "vgg_3_AE_E10_supervised_compare.pdf"
    
    #--------------------------------Bottleneck--------------------------------
    elif tag=="compare_bottleneck":
        #Nicht sonderlich toll der Plot, vllt. eher tabellarisch
        modelidents = (get_path_best_epoch("vgg_3", full_path),
                       get_path_best_epoch("vgg_5_600_picture", full_path),
                       get_path_best_epoch("vgg_5_200", full_path),
                       get_path_best_epoch("vgg_5_64", full_path),
                       get_path_best_epoch("vgg_5_32-eps01", full_path),)
        dataset_array = ["xzt",] * len(modelidents)
        title_of_plot='Accuracy of autoencoders with different bottleneck sizes'
        label_array=["2000", "600", "200", "64", "32"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "vgg_5_"+tag+".pdf"
        y_lims=(0.75,0.95)
        
    elif tag=="compare_600":
        #morefilter and picture (evtl channel)
        modelidents = (get_path_best_epoch("vgg_5_600_picture", full_path),
                       get_path_best_epoch("vgg_5_600_morefilter", full_path),)
        dataset_array = ["xzt",] * len(modelidents)
        title_of_plot='Accuracy of encoders with bottleneck 600'
        label_array=["Picture", "More filter"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "vgg_5_"+tag+".pdf"
        
    elif tag=="compare_200":
        modelidents = (get_path_best_epoch("vgg_5_200", full_path),
                       get_path_best_epoch("vgg_5_200_dense", full_path),)
        dataset_array = ["xzt",] * len(modelidents)
        title_of_plot='Accuracy of encoders with bottleneck 200'
        label_array=["Standard", "Dense"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "vgg_5_"+tag+".pdf"
        
    
    #--------------------------- 200 size variation ---------------------------
    elif tag=="compare_200_smaller":
        modelidents = (get_path_best_epoch("vgg_5_200", full_path),
                       get_path_best_epoch("vgg_5_200_small", full_path),
                       get_path_best_epoch("vgg_5_200_shallow", full_path),)
        dataset_array = ["xzt",] * len(modelidents)
        title_of_plot='Accuracy of encoders with bottleneck 200'
        label_array=["Standard", "Smaller", "Shallower"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "vgg_5_"+tag+".pdf"
        
    elif tag=="compare_200_bigger":
        modelidents = (get_path_best_epoch("vgg_5_200", full_path),
                       get_path_best_epoch("vgg_5_200_large", full_path),
                       get_path_best_epoch("vgg_5_200_deep", full_path),)
        dataset_array = ["xzt",] * len(modelidents)
        title_of_plot='Accuracy of encoders with bottleneck 200'
        label_array=["Standard", "Wider", "Deeper"]
        #in the results/plots/updown_evalutaion/ folder
        plot_file_name = "vgg_5_"+tag+".pdf"
        y_lims=(0.8,0.95)
      
        
    #--------------------------- unfreeze comparison ---------------------------
    elif tag=="compare_unfreeze":
        modelidents = (get_path_best_epoch("vgg_5_200", full_path),
                       get_path_best_epoch("vgg_5_200-unfreeze_contE20", full_path), 
                       get_path_best_epoch("vgg_3_unf", full_path))
        dataset_array = ["xzt",] * len(modelidents)
        title_of_plot='Comparison of partially unfrozen network performance'
        label_array=["Frozen", "Three unfrozen layers", "Supervised network"]
        plot_path="/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/unfreeze/"
        plot_file_name = "vgg_5_"+tag+".pdf"
        
    

    else: NameError("Tag "+tag+" unknown.")
    

    modelidents=[modelpath+modelident for modelident in modelidents]
    save_plot_as = plot_path + plot_file_name

    return modelidents, class_type, dataset_array, title_of_plot, label_array, save_plot_as, plot_type, bins, y_lims



def make_evaluation(tag):
    modelidents, class_type, dataset_array, title_of_plot, label_array, save_plot_as, plot_type, bins, y_lims = get_saved_plots_info(tag)
    
    #generate or load data automatically:
    hist_data_array = make_or_load_files(modelidents, dataset_array, bins, class_type)
    
    print("\n   STATS   ")
    for i,hist_from_model in enumerate(hist_data_array):
        print("Model:", label_array[i])
        print("Mean over all bins:", np.mean(hist_from_model[1]))
    print("\n")
    
    #make plot of multiple data:
    if plot_type == "acc":
        y_label_of_plot="Accuracy"
        fig = make_binned_data_plot(hist_data_array, label_array, title_of_plot, y_label=y_label_of_plot, y_lims=y_lims) 
        plt.show(fig)
        fig.savefig(save_plot_as)
        
    elif plot_type == "loss":
        y_label_of_plot="Loss"
        fig = make_binned_data_plot(hist_data_array, label_array, title_of_plot, y_label=y_label_of_plot) 
        plt.show(fig)
        fig.savefig(save_plot_as)
        
    elif plot_type == None:
        print("plot_type==None: Not generating plots")
    else:
        print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")
    
    print("Plot saved to", save_plot_as)

make_evaluation(tag)



