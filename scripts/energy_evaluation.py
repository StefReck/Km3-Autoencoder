# -*- coding: utf-8 -*-
"""
Take a model that predicts energy of events and do the evaluation for that, either
in the form of a 2d histogramm (mc energy vs reco energy), 
or as a 1d histogram (mc_energy vs mean absolute error).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from get_dataset_info import get_dataset_info
from util.evaluation_utilities import setup_and_make_energy_arr_energy_correct, calculate_2d_hist_data, make_2d_hist_plot, calculate_energy_mae_plot_data, make_energy_mae_plot

samples=None
identifiers = ["2000_unf",]

def get_saved_plots_info(identifier):
    #Info about plots that have been generated for the thesis are listed here.
    dataset_tag="xzt"
    zero_center=True
    energy_bins=np.arange(3,101,1)
    home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"
    
    if identifier=="200_linear":
        model_path="models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_energy_linear_epoch18.h5"
    elif identifier=="2000_unf":
        model_path = "models/vgg_5_2000/trained_vgg_5_2000_supervised_energy_epoch17.h5"
    else:
        raise NameError(identifier+" unknown!")
        
    print("Working on model", model_path)
    
    save_as_base = home_path+"results/plots/energy_evaluation/"+model_path.split("trained_")[1][:-3]
    model_path=home_path+model_path
    return [model_path, dataset_tag, zero_center, energy_bins], save_as_base


def make_or_load_hist_data(model_path, dataset_tag, zero_center, energy_bins, samples=None):
    #Compares the predicted energy and the mc energy of many events in a 2d histogram
    #This function outputs a np array with the 2d hist data, either by loading a saved one, or by
    #generating a new one if no saved one exists.
    #Also outputs the 1d histogram of mc energy over mae.
 
    #name of the files that the hist data will get dumped to (or loaded from)
    modelname = model_path.split("trained_")[1][:-3]
    dump_path="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/"
    name_of_file_2d= dump_path + modelname + "_" + dataset_tag + "_2dhist_data.npy"
    name_of_file_1d= dump_path + modelname + "_" + dataset_tag + "_mae_data.npy"
    
    arr_energy_correct = None
    
    if os.path.isfile(name_of_file_2d)==True:
        print("Loading existing file of 2d histogram data", name_of_file_2d)
        hist_data_2d = np.load(name_of_file_2d)
    else:
        print("No saved 2d histogram data for this model found. New one will be generated.\nGenerating energy array...")
        dataset_info_dict = get_dataset_info(dataset_tag)
        arr_energy_correct = setup_and_make_energy_arr_energy_correct(model_path, dataset_info_dict, zero_center, samples)
        print("Generating 2d histogram...")
        hist_data_2d = calculate_2d_hist_data(arr_energy_correct, energy_bins)
        print("Saving 2d histogram data as", name_of_file_2d)
        np.save(name_of_file_2d, hist_data_2d)
    print("Done.")
    
    
    if os.path.isfile(name_of_file_1d)==True:
        print("Loading existing file of mae data", name_of_file_1d)
        energy_mae_plot_data = np.load(name_of_file_1d)
    else:
        print("No saved mae data for this model found. New one will be generated.\nGenerating energy array...")
        dataset_info_dict = get_dataset_info(dataset_tag)
        if arr_energy_correct == None:
            arr_energy_correct = setup_and_make_energy_arr_energy_correct(model_path, dataset_info_dict, zero_center, samples)
        else:
            print("Energy array from before is reused.")
        print("Generating mae histogramm...")
        energy_mae_plot_data = calculate_energy_mae_plot_data(arr_energy_correct, energy_bins)
        
        print("Saving mae histogram data as", name_of_file_1d)
        np.save(name_of_file_1d, energy_mae_plot_data)
        
    print("Done.")
    return(hist_data_2d, energy_mae_plot_data)


for identifier in identifiers:
    input_for_make_hist_data, save_as_base = get_saved_plots_info(identifier)
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
    fig_mae = make_energy_mae_plot(energy_mae_plot_data)
    
    plt.show(fig_mae)
    if save_as_1d != None:
        print("Saving plot as", save_as_1d)
        fig_mae.savefig(save_as_1d)
        print("Done.")




