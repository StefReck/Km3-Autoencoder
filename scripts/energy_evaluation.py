# -*- coding: utf-8 -*-
"""
Take a model that predicts energy of events and do the evaluation for that.
"""
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
#import pickle

from get_dataset_info import get_dataset_info
from util.evaluation_utilities import make_performance_array_energy_energy, calculate_2d_hist, make_2d_hist_plot
from util.run_cnn import load_zero_center_data, h5_get_number_of_rows

def get_saved_plots_info(identifier):
    dataset_tag="xzt"
    zero_center=True
    energy_bins=np.arange(3,101,1)
    model_folder_path="/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    
    if identifier=="200_linear":
        model_path="vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_energy_linear_epoch18.h5"
    elif identifier="2000_unf":
        model_path = "vgg_5_2000/trained_vgg_5_2000_supervised_energy_epoch17.h5"
    else:
        raise NameError(identifier+" unknown!")
        
    print("Working on model", model_path)
    model_path=model_folder_path+model_path
    return model_path, dataset_tag, zero_center, energy_bins

def make_or_load_2d_hist_data(model_path, dataset_tag, zero_center, energy_bins, samples=None):
    #Compares the predicted energy and the mc energy of many events in a 2d histogram
    #This function outputs a np array with the 2d hist data, either by loading a saved one, or by
    #generating a new one if no saved one exists.
    
    #name of the file that the 2d hist data will get dumped to (or loaded from)
    modelname = model_path.split("trained_")[1][:-3]
    dump_path="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/"
    name_of_file= dump_path + modelname + "_" + dataset_tag + "_2dhist_data.npy"
            
    #The model that does the prediction
    model=load_model(model_path)
    #The dataset to be used to predict on; the prediction is done on the test file
    dataset_info_dict = get_dataset_info(dataset_tag)
    #home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    #broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
    filesize_factor=dataset_info_dict["filesize_factor"]
    #filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
        
    #Zero-Center with precalculated mean image
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    #test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    n_gpu=(1, 'avolkov')
    if zero_center == True:
        xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=n_gpu[0])
    else:
        xs_mean = None
    
    
    if os.path.isfile(name_of_file)==True:
        print("Opening existing file of histogram data", name_of_file)
        hist_data_2d = np.load(name_of_file)
        #with open(name_of_file, "rb") as dump_file:
        #    hist_data_2d = pickle.load(dump_file)
            
    else:
        print("No saved histogram data for this model found. New one will be generated.\nGenerating energy array...")
        arr_energy_correct = make_performance_array_energy_energy(model, test_file, [1,"energy"], 
                                                                  xs_mean, None, dataset_info_dict, samples)
        print("Generating histogram...")
        hist_data_2d = calculate_2d_hist(arr_energy_correct, energy_bins)
        
        print("Saving histogram data as", name_of_file)
        np.save(name_of_file, hist_data_2d)
        #with open(name_of_file, "wb") as dump_file:
        #    pickle.dump(hist_data_2d, dump_file)
        
    print("Done.")
    return(hist_data_2d)
    
input_for_make_hist_data = get_saved_plots_info("200_linear")
hist_data_2d = make_or_load_2d_hist_data(*input_for_make_hist_data)
fig = make_2d_hist_plot(hist_data_2d)
plt.show(fig)

