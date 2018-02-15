# -*- coding: utf-8 -*-

"""
Evaluate model performance after training. 
This is for comparison of supervised accuracy on different datasets.
Especially fro the plots for the broken data comparison.
"""

import numpy as np
from keras.models import load_model
import pickle
import os

from util.evaluation_utilities import *
from util.run_cnn import load_zero_center_data, h5_get_number_of_rows
from get_dataset_info import get_dataset_info


# "broken_data_mode_"+enc oder +unf
#e.g. "1enc"
which_one="2unf"
#instead of plotting acc vs. energy, one can also make a compare plot, which shows the difference
#between "on simulations" and "on measured data"
make_difference_plot=True


def get_info(which_one):
    if which_one=="1unf":
        #vgg_3_broken1_unf
        modelidents = ("vgg_3-broken1/trained_vgg_3-broken1_supervised_up_down_epoch6.h5",
                       "vgg_3-broken1/trained_vgg_3-broken1_supervised_up_down_epoch6.h5",
                       "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Unfrozen network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken1_unf.pdf" 
        #y limits of plot:
        y_lims=(0.4,1.05)
    
    elif which_one=="1enc":
        #vgg_3_broken1_enc
        modelidents = ("vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_epoch14.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_epoch14.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_accdeg_epoch24.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Autoencoder-encoder network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken1_enc.pdf" 
        #y limits of plot:
        y_lims=(0.7,1.0)
    
    elif which_one=="2unf":
        #vgg_3_broken2_unf
        modelidents = ("vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5",
                       "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5",
                       "vgg_3-noise10/trained_vgg_3-noise10_supervised_up_down_epoch6.h5")
        #Which dataset each to use
        dataset_array = ("xzt", "xzt_broken2", "xzt_broken2")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Unfrozen network performance with noisy data'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken2_unf.pdf" 
        #y limits of plot:
        y_lims=(0.4,0.95)
        
    elif which_one=="2enc":
        #vgg_3_broken2_enc
        modelidents = ("vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch10_supervised_up_down_epoch9.h5",
                       "vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch10_supervised_up_down_epoch9.h5",
                       "vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch10_supervised_up_down_noise_epoch14.h5")
        #Which dataset each to use
        dataset_array = ("xzt", "xzt_broken2", "xzt_broken2")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Autoencoder-encoder network performance with noisy data'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken2_enc.pdf" 
        #y limits of plot:
        y_lims=(0.4,0.95)
    
    else:
        print(which_one, "is not known!")
        raise(TypeError)
        
    return modelidents,dataset_array,title_of_plot,plot_file_name,y_lims

modelidents,dataset_array,title_of_plot,plot_file_name,y_lims = get_info(which_one)

label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
#Overwrite default color palette. Leave empty for auto
color_array=["orange", "blue", "navy"]
#loss, acc, None
plot_type = "acc"
#Info about model
class_type = (2, 'up_down')



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"

modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
#           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
for modelident in modelidents:
    modelnames.append(modelident.split("trained_")[1][:-3])
    
save_plot_as = plot_path + plot_file_name
    

#Accuracy as a function of energy binned to a histogramm. It is dumped automatically into the
#results/data folder, so that it has not to be generated again
def make_and_save_hist_data(modelpath, dataset, modelident, class_type, name_of_file):
    model = load_model(modelpath + modelident)
    
    dataset_info_dict = get_dataset_info(dataset)
    #home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"]
    
    train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=32, n_bins=n_bins, n_gpu=1)

    
    print("Making energy_coorect_array of ", modelident)
    arr_energy_correct = make_performance_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, broken_simulations_mode=broken_simulations_mode, swap_4d_channels=None, samples=None, dataset_info_dict=dataset_info_dict)
    #hist_data = [bin_edges_centered, hist_1d_energy_accuracy_bins]:
    hist_data = make_energy_to_accuracy_data(arr_energy_correct, plot_range=(3,100))
    #save to file
    print("Saving hist_data as", name_of_file)
    with open(name_of_file, "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data


"""
#Loss of an AE as a function of energy, rest like above
def make_and_save_hist_data_autoencoder(modelpath, modelident, modelname, test_file, n_bins, class_type, xs_mean):
    model = load_model(modelpath + modelident)
    print("Making and saving energy_loss_array of autoencoder ", modelname)
    hist_data = make_autoencoder_energy_data(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, swap_4d_channels=None, samples=None)
    #save to file
    with open("/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_hist_data.txt", "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data
"""


#open dumped histogramm data, that was generated from the above two functions
def open_hist_data(name_of_file):
    #hist data is list len 2 with 0:energy array, 1:acc/loss array
    print("Opening existing hist_data file", name_of_file)
    #load again
    with open(name_of_file, "rb") as dump_file:
        hist_data = pickle.load(dump_file)
    return hist_data

def make_or_load_files(modelnames, dataset_array, modelidents=None, modelpath=None, class_type=None):
    hist_data_array=[]
    for i,modelname in enumerate(modelnames):
        dataset=dataset_array[i]
        
        print("Working on ",modelname,"using dataset", dataset)
        name_of_file="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_" + dataset + "_hist_data.txt"
        if os.path.isfile(name_of_file)==True:
            hist_data_array.append(open_hist_data(name_of_file))
        else:
            hist_data = make_and_save_hist_data(modelpath, dataset, modelidents[i], class_type, name_of_file)
            hist_data_array.append(hist_data)
        print("Done.")
    return hist_data_array



if make_difference_plot == False:
    #generate or load data automatically:
    hist_data_array = make_or_load_files(modelnames, dataset_array, modelidents, modelpath, class_type)
    #make plot of multiple data:
    if plot_type == "acc":
        y_label_of_plot="Accuracy"
        make_energy_to_accuracy_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array) 
    elif plot_type == "loss":
        y_label_of_plot="Loss"
        make_energy_to_loss_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, color_array=color_array) 
    elif plot_type == None:
        print("plot_type==None: Not generating plots")
    else:
        print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")
    
    print("Plot saved to", save_plot_as)
else:
    #which plots to make diff of; first - second
    make_diff_of_list=((0,1),(2,1))
    title_list=("Loss of accuracy when moving from 'simulations' to 'measured' data",
                "DIfference in accuracy: Upper limit - 'measured' data")
    save_as_list=(plot_path + "vgg_3_broken2_sim_real.pdf", 
                  plot_path + "vgg_3_broken2_upper_real.pdf")
    y_lims_list=((-0.04,0.09),(-0.03,0.06))
    
    for i in range(len(make_diff_of_list)):
        #label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
        modelidents,dataset_array,title_of_plot,plot_file_name,y_lims = get_info("2unf")
        
        modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
        #           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
        for modelident in modelidents:
            modelnames.append(modelident.split("trained_")[1][:-3])
        
        hist_data_array_unf = make_or_load_files(modelnames, dataset_array, modelidents, modelpath, class_type)
        
        
        modelidents,dataset_array,title_of_plot,plot_file_name,y_lims = get_info("2enc")
        
        modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
        #           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
        for modelident in modelidents:
            modelnames.append(modelident.split("trained_")[1][:-3])
            
        hist_data_array_enc = make_or_load_files(modelnames, dataset_array, modelidents, modelpath, class_type)
        
        
        label_array=["Unfrozen", "Autoencoder-encoder"]
        #Overwrite default color palette. Leave empty for auto
        color_array=[]
        #loss, acc, None
        plot_type = "acc"
        #Info about model
        class_type = (2, 'up_down')
      
        modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
        plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"
        
        title_of_plot=title_list[i]
        save_plot_as = save_as_list[i]
        y_lims=y_lims_list[i]
        make_diff_of=make_diff_of_list[i]
        
        hist_data_array_diff=[]
        hist_1=hist_data_array_unf[make_diff_of[0]]
        hist_2=hist_data_array_unf[make_diff_of[1]]
        diff_hist=[hist_1[0], hist_1[1]-hist_2[1]]
        hist_data_array_diff.append(diff_hist)
        
        hist_1=hist_data_array_enc[make_diff_of[0]]
        hist_2=hist_data_array_enc[make_diff_of[1]]
        diff_hist=[hist_1[0], hist_1[1]-hist_2[1]]
        hist_data_array_diff.append(diff_hist)
    
        #make plot of multiple data:
        if plot_type == "acc":
            y_label_of_plot="Difference in accuracy"
            make_energy_to_accuracy_plot_comp_data(hist_data_array_diff, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array) 
        elif plot_type == "loss":
            y_label_of_plot="Loss"
            make_energy_to_loss_plot_comp_data(hist_data_array_diff, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, color_array=color_array) 
        elif plot_type == None:
            print("plot_type==None: Not generating plots")
        else:
            print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")
        
        print("Plot saved to", save_plot_as)
