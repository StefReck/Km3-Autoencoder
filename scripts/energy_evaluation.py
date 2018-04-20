# -*- coding: utf-8 -*-
"""
Take a model that predicts energy of events and do the evaluation for that, either
in the form of a 2d histogramm (mc energy vs reco energy), 
or as a 1d histogram (mc_energy vs mean absolute error).
"""
import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Take a model that predicts energy of events and do the evaluation for that, either in the form of a 2d histogramm (mc energy vs reco energy), or as a 1d histogram (mc_energy vs mean absolute error).')
    parser.add_argument('model', type=str, help='Name of a model .h5 file, or a identifier for a saved setup.')

    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
identifier = params["model"]

import matplotlib.pyplot as plt
import numpy as np
import os, sys

from get_dataset_info import get_dataset_info
from util.evaluation_utilities import setup_and_make_energy_arr_energy_correct, calculate_2d_hist_data, make_2d_hist_plot, calculate_energy_mae_plot_data, make_energy_mae_plot

#Which model to use (see below)
#identifiers = ["2000_unf",]

#only go through parts of the file (for testing)
samples=None

def get_saved_plots_info(identifier):
    #Info about plots that have been generated for the thesis are listed here.
    dataset_tag="xzt"
    zero_center=True
    energy_bins_2d=np.arange(3,101,1)
    energy_bins_1d=np.linspace(3,100,32)
    home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"
    
    if identifier=="200_linear":
        model_path="models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_energy_linear_epoch18.h5"
    elif identifier=="2000_unf":
        model_path = "models/vgg_5_2000/trained_vgg_5_2000_supervised_energy_epoch17.h5"
    elif identifier=="2000_unf_mse":
        model_path = "models/vgg_5_2000-mse/trained_vgg_5_2000-mse_supervised_energy_epoch10.h5"
        
    #----------------Sets for mae comparison----------------
    # Will exit after completion
    elif identifier == "2000":
        identifiers = ["2000_unf", "2000_unf_mse"]
        label_list  = ["Optimized for mean absolute error", "Optimized for mean squared error"]
        save_plot_as = home_path+"results/plots/energy_evaluation/mae_compare_set_"+identifier+"_plot.pdf"
        compare_plots(identifiers, label_list, save_plot_as)
    elif identifier == "bottleneck":
        identifiers = ["2000_unf", "200_linear"]
        label_list  = ["Unfrozen 2000", "Encoder 200"]
        save_plot_as = home_path+"results/plots/energy_evaluation/mae_compare_set_"+identifier+"_plot.pdf"
        compare_plots(identifiers, label_list, save_plot_as)
    #-------------------------------------------------------
        
    else:
        print("Input is not a known identifier. Opening as model instead.")
        model_path = identifier
        save_as_base = home_path+"results/plots/energy_evaluation/"+model_path.split("trained_")[1][:-3]
        return [model_path, dataset_tag, zero_center, energy_bins_2d, energy_bins_1d], save_as_base
        
    print("Working on model", model_path)
    
    save_as_base = home_path+"results/plots/energy_evaluation/"+model_path.split("trained_")[1][:-3]
    model_path=home_path+model_path
    return [model_path, dataset_tag, zero_center, energy_bins_2d, energy_bins_1d], save_as_base

def get_dump_names(model_path, dataset_tag):
    #Returns the name of the saved statistics files
    modelname = model_path.split("trained_")[1][:-3]
    dump_path="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/"
    name_of_file_2d= dump_path + "energy_" + modelname + "_" + dataset_tag + "_2dhist_data.npy"
    name_of_file_1d= dump_path + "energy_" + modelname + "_" + dataset_tag + "_mae_data.npy"
    return name_of_file_1d, name_of_file_2d

def make_or_load_hist_data(model_path, dataset_tag, zero_center, energy_bins_2d, energy_bins_1d, samples=None):
    #Compares the predicted energy and the mc energy of many events in a 2d histogram
    #This function outputs a np array with the 2d hist data, either by loading a saved one, or by
    #generating a new one if no saved one exists.
    #Also outputs the 1d histogram of mc energy over mae.
 
    #name of the files that the hist data will get dumped to (or loaded from)
    name_of_file_1d, name_of_file_2d = get_dump_names(model_path, dataset_tag)
    
    arr_energy_correct = []
    
    if os.path.isfile(name_of_file_2d)==True:
        print("Loading existing file of 2d histogram data", name_of_file_2d)
        hist_data_2d = np.load(name_of_file_2d)
    else:
        print("No saved 2d histogram data for this model found. New one will be generated.\nGenerating energy array...")
        dataset_info_dict = get_dataset_info(dataset_tag)
        arr_energy_correct = setup_and_make_energy_arr_energy_correct(model_path, dataset_info_dict, zero_center, samples)
        print("Generating 2d histogram...")
        hist_data_2d = calculate_2d_hist_data(arr_energy_correct, energy_bins_2d)
        print("Saving 2d histogram data as", name_of_file_2d)
        np.save(name_of_file_2d, hist_data_2d)
    print("Done.")
    
    
    if os.path.isfile(name_of_file_1d)==True:
        print("Loading existing file of mae data", name_of_file_1d)
        energy_mae_plot_data = np.load(name_of_file_1d)
    else:
        print("No saved mae data for this model found. New one will be generated.\nGenerating energy array...")
        dataset_info_dict = get_dataset_info(dataset_tag)
        if len(arr_energy_correct) == 0:
            arr_energy_correct = setup_and_make_energy_arr_energy_correct(model_path, dataset_info_dict, zero_center, samples)
        else:
            print("Energy array from before is reused.")
        print("Generating mae histogramm...")
        energy_mae_plot_data = calculate_energy_mae_plot_data(arr_energy_correct, energy_bins_1d)
        
        print("Saving mae histogram data as", name_of_file_1d)
        np.save(name_of_file_1d, energy_mae_plot_data)
        
    print("Done.")
    return(hist_data_2d, energy_mae_plot_data)


def save_and_show_plots(identifier):
    #Main function. Generate or load the data for the plots, and make them.
    input_for_make_hist_data, save_as_base = get_saved_plots_info(identifier)
    #This function ill exit after completion if a set was chosen
    
    save_as_2d = save_as_base+"_2dhist_plot.pdf"
    save_as_1d = save_as_base+"_mae_plot.pdf"
        
    
    hist_data_2d, energy_mae_plot_data = make_or_load_hist_data(*input_for_make_hist_data, samples=samples)
    
    print("Generating hist2d plot...")
    fig_hist2d = make_2d_hist_plot(hist_data_2d)
    
    plt.show(fig_hist2d)
    if save_as_2d != None:
        print("Saving plot as", save_as_2d)
        fig_hist2d.savefig(save_as_2d)
        print("Done.")
        
    
    print("Generating mae plot...")
    fig_mae = make_energy_mae_plot([energy_mae_plot_data,])
    
    plt.show(fig_mae)
    if save_as_1d != None:
        print("Saving plot as", save_as_1d)
        fig_mae.savefig(save_as_1d)
        print("Done.")

def compare_plots(identifiers, label_list, save_plot_as):
    """
    Plot several saved mae data files and plot them in a single figure.
    """
    mae_plot_data_list = []
    print("Loading the saved files of the following models:")
    for identifier in identifiers:
        [model_path, dataset_tag, zero_center, energy_bins_2d, energy_bins_1d], save_as_base = get_saved_plots_info(identifier)
        name_of_file_1d, name_of_file_2d = get_dump_names(model_path, dataset_tag)
        
        mae_plot_data = np.load(name_of_file_1d)
        mae_plot_data_list.append(mae_plot_data)

    print("Done. Generating plot...")
    fig_mae = make_energy_mae_plot(mae_plot_data_list, label_list=label_list)
    print("Saving plot as", save_plot_as)
    fig_mae.savefig(save_plot_as)
    plt.show(fig_mae)
    sys.exit()
    
    
save_and_show_plots(identifier)

   

