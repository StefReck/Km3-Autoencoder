# -*- coding: utf-8 -*-
"""
Take a model that predicts energy of events and do the evaluation for that.
"""
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from get_dataset_info import get_dataset_info
from util.evaluation_utilities import make_performance_array_energy_energy, calculate_2d_hist, make_2d_hist_plot
from util.run_cnn import load_zero_center_data, h5_get_number_of_rows

model_path="/home/woody/capn/mppi013h/Km3-Autoencoder/models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_energy_linear_epoch18.h5"
dataset_tag="xzt"
zero_center=True
energy_bins=np.arange(3,101,1)



#name of the file that the 2d hist data will get dumped to (or loaded from)
modelname = model_path.split("trained_")[1][:-3]
dump_path="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/"
name_of_file= dump_path + modelname + "_" + dataset_tag + "_2dhist_data.txt"
        


model=load_model(model_path)
dataset_info_dict = get_dataset_info(dataset_tag)
home_path=dataset_info_dict["home_path"]
train_file=dataset_info_dict["train_file"]
test_file=dataset_info_dict["test_file"]
n_bins=dataset_info_dict["n_bins"]
broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
filesize_factor=dataset_info_dict["filesize_factor"]
filesize_factor_test=dataset_info_dict["filesize_factor_test"]
batchsize=dataset_info_dict["batchsize"] #def 32
    
#fit_model and evaluate_model take lists of tuples, so that you can give many single files (doesnt work?)
train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]

#Zero-Center with precalculated mean image
n_gpu=(1, 'avolkov')
if zero_center == True:
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=n_gpu[0])
else:
    xs_mean = None


if os.path.isfile(name_of_file)==True:
    print("Opening exiting file", name_of_file)
    with open(name_of_file, "rb") as dump_file:
        hist_data_2d = pickle.load(dump_file)
else:
    print("Generating energy array...")
    arr_energy_correct = make_performance_array_energy_energy(model, test_file, [1,"energy"], 
                                                              xs_mean, None, dataset_info_dict)
    print("Generating histogram...")
    hist_data_2d = calculate_2d_hist(arr_energy_correct, energy_bins)
    
    print("Saving histogram data as", name_of_file)
    with open(name_of_file, "wb") as dump_file:
        pickle.dump(hist_data_2d, dump_file)

print("Done. Generating plot...")
fig = make_2d_hist_plot(hist_data_2d)
plt.show(fig)

