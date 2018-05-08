# -*- coding: utf-8 -*-
"""
    Evalutaion for energy regression models
    
Take a model that predicts energy of events and do the evaluation for that, both
in the form of a 2d histogramm (mc energy vs reco energy), 
and as a 1d histogram (mc_energy vs mean absolute error).

Looks for saved arr_energ_corrects to load, or generate new one.
Will print statistics from that array like median, variance, ...
Generates the 2d and the 1d plots and saves them.

Can also compare multiple 1d plots instead.

If apply precuts is selected, the dataset will be xzt_precuts instead of xzt, 
and _precut will be added to the filename and the saved arr_energy_correct.
"""
import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Take a model that predicts energy of events and do the evaluation for that, either in the form of a 2d histogramm (mc energy vs reco energy), or as a 1d histogram (mc_energy vs mean absolute error).')
    parser.add_argument('model', type=str, help='Name of a model .h5 file, or an identifier for a saved setup. (see this file for identifiers for sets and saved_setups for single epochs)')
    parser.add_argument('-p','--apply_precuts', help="Change to dataset xzt_precut", action='store_true')

    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()

import matplotlib.pyplot as plt
import numpy as np
import os

from get_dataset_info import get_dataset_info
from util.evaluation_utilities import setup_and_make_energy_arr_energy_correct, calculate_2d_hist_data, make_2d_hist_plot, calculate_energy_mae_plot_data, make_energy_mae_plot, make_energy_evaluation_statistics
from util.saved_setups_for_plot_statistics import get_path_best_epoch


identifier = params["model"]
#Should precuts be applied to the data; if so, the plot will be saved 
#with a "_precut" added to the file name
apply_precuts = params["apply_precuts"]
#only go through parts of the file (for testing)
samples=None


def get_saved_plots_info(identifier, apply_precuts=False):
    #Info about plots that have been generated for the thesis are listed here.
    dataset_tag="xzt"
    zero_center=True
    energy_bins_2d=np.arange(3,101,1)
    energy_bins_1d=np.linspace(3,100,32)
    home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"
    is_a_set=False
    
    
    #----------------Sets for mae comparison----------------
    if identifier == "2000":
        identifiers = ["2000_unf", "2000_unf_mse"]
        label_list  = ["With MAE", "With MSE"]
        save_plot_as = home_path+"results/plots/energy_evaluation/mae_compare_set_"+identifier+"_plot.pdf"
        is_a_set=True
    elif identifier == "bottleneck":
        identifiers = ["2000_unf", "200_linear"]
        label_list  = ["Unfrozen 2000", "Encoder 200"]
        save_plot_as = home_path+"results/plots/energy_evaluation/mae_compare_set_"+identifier+"_plot.pdf"
        is_a_set=True
    #-------------------------------------------------------
        
    else:
        try:
            #Read in the saved name from saved_setups_for_plot_statistics
            model_path = get_path_best_epoch(identifier, full_path=False)
        except NameError:
            print("Input is not a known identifier. Opening as model instead.")
            model_path = identifier
            save_as_base = home_path+"results/plots/energy_evaluation/"+model_path.split("trained_")[1][:-3]
            return [model_path, dataset_tag, zero_center, energy_bins_2d, energy_bins_1d], save_as_base
        

    if is_a_set:
        return [identifiers, label_list], save_plot_as
    else:
        print("Working on model", model_path)
        #Where to save the plots to
        save_as_base = home_path+"results/plots/energy_evaluation/"+model_path.split("trained_")[1][:-3]
        if apply_precuts:
            save_as_base+="precut_"
            dataset_tag="xzt_precut"
        
        model_path=home_path+"models/"+model_path
        return [model_path, dataset_tag, zero_center, energy_bins_2d, energy_bins_1d], save_as_base


def get_dump_name_arr(model_path, dataset_tag):
    #Returns the name and path of the energy correct array dump file
    modelname = model_path.split("trained_")[1][:-3]
    dump_path="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/"
    
    name_of_arr = dump_path + "energy_" + modelname + "_" + dataset_tag + "_arr_correct.npy"
        
    return name_of_arr


def make_or_load_hist_data(model_path, dataset_tag, zero_center, energy_bins_2d, energy_bins_1d, samples=None):
    #Compares the predicted energy and the mc energy of many events in a 2d histogram
    #This function outputs a np array with the 2d hist data, 
    #either by loading a saved arr_energy_correct, or by generating a new one
    #Also outputs the 1d histogram of mc energy over mae.
 
    #name of the files that the hist data will get dumped to (or loaded from)
    name_of_arr = get_dump_name_arr(model_path, dataset_tag)

    if os.path.isfile(name_of_arr)==True:
        print("Loading existing file of correct array", name_of_arr)
        arr_energy_correct = np.load(name_of_arr)
        #Print infos about the evaluation performance like Median, Variance,...
        __ = make_energy_evaluation_statistics(arr_energy_correct)
    else:
        print("No saved correct array for this model found. New one will be generated.\nGenerating energy array...")
        dataset_info_dict = get_dataset_info(dataset_tag)
        arr_energy_correct = setup_and_make_energy_arr_energy_correct(model_path, dataset_info_dict, zero_center, samples)
        print("Saving as", name_of_arr)
        np.save(name_of_arr, arr_energy_correct)
        
    print("Generating 2d histogram...")
    hist_data_2d = calculate_2d_hist_data(arr_energy_correct, energy_bins_2d)
    print("Done.")
    
    print("Generating mae histogramm...")
    energy_mae_plot_data = calculate_energy_mae_plot_data(arr_energy_correct, energy_bins_1d)
    print("Done.")
    
    return(hist_data_2d, energy_mae_plot_data)


def save_and_show_plots(identifier, apply_precuts=False):
    #Main function. Generate or load the data for the plots, and make them.
    input_for_make_hist_data, save_as_base = get_saved_plots_info(identifier, apply_precuts)
    
    #Can also compare multiple already generated mae plots
    if len(input_for_make_hist_data)==2:
        #Compare existing mae plots
        save_plot_as = save_as_base
        fig_compare = compare_plots(*input_for_make_hist_data)
        if save_plot_as != None:
            print("Saving plot as", save_plot_as)
            fig_compare.savefig(save_plot_as)
            print("Done")
        plt.show(fig_compare)
        
    else:    
        #Do the standard energy evaluation
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


def compare_plots(identifiers, label_list, apply_precuts=False):
    """
    Plot several saved mae data files and plot them in a single figure.
    """
    mae_plot_data_list = []
    print("Loading the saved files of the following models:")
    for identifier in identifiers:
        input_for_make_hist_data, save_as_base = get_saved_plots_info(identifier, apply_precuts)
        hist_data_2d, mae_plot_data = make_or_load_hist_data(*input_for_make_hist_data)
        mae_plot_data_list.append(mae_plot_data)

    print("Done. Generating plot...")
    fig_mae = make_energy_mae_plot(mae_plot_data_list, label_list=label_list)
    return fig_mae
    
    
save_and_show_plots(identifier, apply_precuts)

   

