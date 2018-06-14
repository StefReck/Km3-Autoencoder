# -*- coding: utf-8 -*-

"""
Evaluate model performance after training. 
This is for comparison of supervised accuracy on different datasets.
Especially for the plots for the broken data comparison.

The usual setup is that simulations are broken (so that the AE has not to be trained again)
so 3 tests are necessary:
    Trained on broken --> Test on broken (seeming performance)
    Trained on broken --> Test on real   (actual perfromance)
    Trained on real   --> Test on real   (best case)
    
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from util.evaluation_utilities import make_or_load_files, make_binned_data_plot, make_energy_mae_plot_mean_only, make_energy_mae_plot_mean_only_single
from util.saved_setups_for_plot_statistics import get_path_best_epoch
from energy_evaluation import make_or_load_hist_data

def parse_input():
    parser = argparse.ArgumentParser(description='Evaluate model performance after training. This is for comparison of supervised accuracy on different datasets. Especially for the plots for the broken data comparison.')
    parser.add_argument('info_tags', nargs="+", type=str, help='Names of identifiers for a saved setup. All for making all available ones.')

    args = parser.parse_args()
    params = vars(args)
    return params

#Standard, plot acc vs energy plots of these saved setups (taken from parser now)
#which_ones=("4_64_enc",)


#extra string to be included in file names
extra_name=""
#number of bins of the histogram plot; default (is 97) is 32 now; backward compatibility with 98 bins
bins=32

#If not None: Change the y range of all plots to this one (to make unit looks)
y_lims_override = None

#instead of plotting acc vs. energy, one can also make a compare plot, 
#which shows the difference #between "on simulations" and "on measured data"
#then, the number of the broken mode has to be given
#can be True, False or "both"
#TODO Rework, disfunctional
make_difference_plot=False
which_broken_study=4

def get_procedure(broken_model, real_model, brokendata_tag, realdata_tag):
    #For when the "Simulation"-dataset is manipulated simulations:
    modelidents   = (broken_model,   broken_model, real_model)
    dataset_array = (brokendata_tag, realdata_tag, realdata_tag)
    return modelidents, dataset_array

def get_info(which_one, extra_name="", y_lims_override=None):
    """
    Saved setups of plots. 
    Returns all relevant infos to exactly produce (or reproduce) these plots.
    """
    #DEFAULT VALUES (overwritten when necessary)
    #This will be added before all modelidents
    modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    #Default class type the evaluation is done for. None for autoencoders.
    class_type = (2, 'up_down')
    #mse, acc, mre
    plot_type = "acc"
    #Default location of legend ("best")
    legend_loc="best"
    #ylims of plot ( only for acc )
    y_lims=(0.5,1.0)
    #Where to save the plots
    plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"
    folder_in_the_plots_path = "broken_study/"
    
    #Labels for the plot are defined below now!
    #label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
    title_of_plot=""
    #Overwrite default color palette. Leave empty for auto
    color_array=["orange", "blue", "navy"]
    
    
    #Add the number of bins to the name of the plot file (usually 32)
    extra_name="_"+ str(bins)+"_bins" + extra_name
    
    try: which_one=int(which_one)
    except: ValueError
    # ----------------------------- Up down -----------------------------
    if which_one=="1_unf" or which_one==0:
        #vgg_3_broken1_unf
        modelidents = ("vgg_3-broken1/trained_vgg_3-broken1_supervised_up_down_epoch6.h5",
                       "vgg_3-broken1/trained_vgg_3-broken1_supervised_up_down_epoch6.h5",
                       "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Unfrozen network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken1_unf"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.4,1.05)
        
    
    elif which_one=="1_enc" or which_one==1:
        #vgg_3_broken1_enc
        modelidents = ("vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_epoch14.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken1_epoch14.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_accdeg_epoch24.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Autoencoder-encoder network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken1_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.4,1.05)
        legend_loc="lower right"
    
    elif which_one=="2_unf" or which_one==2:
        #vgg_3_broken2_unf
        modelidents = ("vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5",
                       "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5",
                       "vgg_3-noise10/trained_vgg_3-noise10_supervised_up_down_epoch6.h5")
        #Which dataset each to use
        dataset_array = ("xzt", "xzt_broken2", "xzt_broken2")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Unfrozen network performance with noisy data'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken2_unf"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.68,0.96)
        legend_loc="lower right"
        
    elif which_one=="2_enc" or which_one==3:
        #vgg_3_broken2_enc
        modelidents = ("vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch10_supervised_up_down_epoch9.h5",
                       "vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch10_supervised_up_down_epoch9.h5",
                       "vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch10_supervised_up_down_noise_epoch14.h5")
        #Which dataset each to use
        dataset_array = ("xzt", "xzt_broken2", "xzt_broken2")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Autoencoder-encoder network performance with noisy data'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken2_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.68,0.96)
        legend_loc="lower right"
    
    elif which_one=="4_unf" or which_one==4:
        modelidents = ("vgg_3-broken4/trained_vgg_3-broken4_supervised_up_down_epoch4.h5",
                       "vgg_3-broken4/trained_vgg_3-broken4_supervised_up_down_epoch4.h5",
                       "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Unfrozen network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken4_unf"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.5,1.0)
    
    elif which_one=="4_enc" or which_one==5:
        modelidents = ("vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken4_epoch52.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken4_epoch52.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_accdeg_epoch24.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Autoencoder-encoder network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken4_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.5,1.0)
        
    elif which_one=="4_pic_enc" or which_one==6:
        modelidents = ("vgg_5_picture/trained_vgg_5_picture_autoencoder_epoch48_supervised_up_down_broken4_epoch53.h5",
                       "vgg_5_picture/trained_vgg_5_picture_autoencoder_epoch48_supervised_up_down_broken4_epoch53.h5",
                       "vgg_5_picture/trained_vgg_5_picture_autoencoder_epoch48_supervised_up_down_epoch74.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='600 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_picture_broken4_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,0.95)
    
    elif which_one=="4_200_enc" or which_one==7:
        modelidents = ("vgg_5_200/trained_vgg_5_200_autoencoder_epoch94_supervised_up_down_broken4_epoch59.h5",
                       "vgg_5_200/trained_vgg_5_200_autoencoder_epoch94_supervised_up_down_broken4_epoch59.h5",
                       "vgg_5_200/trained_vgg_5_200_autoencoder_epoch94_supervised_up_down_epoch45.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='200 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_200_broken4_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,0.95)
    
    elif which_one=="4_64_enc" or which_one==8:
        modelidents = ("vgg_5_64/trained_vgg_5_64_autoencoder_epoch64_supervised_up_down_broken4_epoch57.h5",
                       "vgg_5_64/trained_vgg_5_64_autoencoder_epoch64_supervised_up_down_broken4_epoch57.h5",
                       "vgg_5_64/trained_vgg_5_64_autoencoder_epoch64_supervised_up_down_epoch26.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='64 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_64_broken4_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,0.95)
        
    elif which_one=="4_64_enc_nodrop" or which_one==26:
        modelidents = ("vgg_5_64/trained_vgg_5_64_autoencoder_epoch82_supervised_up_down_broken4_nodrop_epoch52.h5",
                       "vgg_5_64/trained_vgg_5_64_autoencoder_epoch82_supervised_up_down_broken4_nodrop_epoch52.h5",
                       "vgg_5_64/trained_vgg_5_64_autoencoder_epoch64_supervised_up_down_nodrop_epoch69.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='64 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_64_broken4_enc_nodrop"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,0.95)
    
    elif which_one=="4_32_enc" or which_one==9:
        modelidents = ("vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch31_supervised_up_down_broken4_epoch1.h5",
                       "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch31_supervised_up_down_broken4_epoch1.h5",
                       "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch31_supervised_up_down_epoch48.h5")
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        title_of_plot='32 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_32_broken4_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,0.95)
        legend_loc="lower right"
    elif which_one=="4_32_enc_nodrop" or which_one==23:
        modelidents = ("vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch22_supervised_up_down_broken4_nodrop_epoch47.h5",
                       "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch22_supervised_up_down_broken4_nodrop_epoch47.h5",
                       "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch31_supervised_up_down_nodrop_epoch79.h5")
        dataset_array = ("xzt_broken4", "xzt", "xzt")
        title_of_plot='32 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_32_broken4_enc_nodrop"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,0.95)
        legend_loc="lower right"    
    
    
    elif which_one=="4flip_unf" or which_one==10:
        modelidents = ("vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5",
                       "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5",
                       "vgg_3-broken4/trained_vgg_3-broken4_supervised_up_down_epoch4.h5")
        #Which dataset each to use
        dataset_array = ("xzt", "xzt_broken4", "xzt_broken4")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Unfrozen network performance with manipulated data'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken4_flip_unf"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.75,1.0)
    elif which_one=="4flip_enc" or which_one==11:
        modelidents = ("vgg_3-broken4/trained_vgg_3-broken4_autoencoder_epoch12_supervised_up_down_xzt_epoch62.h5",
                       "vgg_3-broken4/trained_vgg_3-broken4_autoencoder_epoch12_supervised_up_down_xzt_epoch62.h5",
                       "vgg_3-broken4/trained_vgg_3-broken4_autoencoder_epoch10_supervised_up_down_broken4_epoch59.h5")
        #Which dataset each to use
        dataset_array = ("xzt", "xzt_broken4", "xzt_broken4")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Autoencoder-encoder network performance with manipulated data'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken4_flip_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.75,1)
        
        
    elif which_one=="5_enc" or which_one==12:
        modelidents = ("vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken5_epoch58.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_broken5_epoch58.h5",
                       "vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_accdeg_epoch24.h5")
        #Which dataset each to use
        dataset_array = ("xzt_broken5", "xzt", "xzt")
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Autoencoder-encoder network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken5_enc"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,1.0)
        legend_loc="lower right"
        
    elif which_one=="5_unf" or which_one==13:
        broken_model = "vgg_3-broken5/trained_vgg_3-broken5_supervised_up_down_epoch6.h5"
        real_model   = "vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5"
        brokendata_tag = "xzt_broken5"
        realdata_tag   = "xzt"
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Unfrozen network performance with manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_3_broken5_unf"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,1.0)
        legend_loc="lower right"
    
    elif which_one=="4_200_large_enc" or which_one==14:
        broken_model = "vgg_5_200_large/trained_vgg_5_200_large_autoencoder_epoch39_supervised_up_down_broken4_epoch34.h5"
        real_model   = get_path_best_epoch("vgg_5_200_large", full_path=False)
        brokendata_tag = "xzt_broken4"
        realdata_tag   = "xzt"
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        title_of_plot='Large 200 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_200_large_broken4_enc"+extra_name+".pdf" 
        y_lims=(0.7,0.95)
    elif which_one=="4_200_small_enc" or which_one==15:
        broken_model = "vgg_5_200_small/trained_vgg_5_200_small_autoencoder_epoch77_supervised_up_down_broken4_epoch57.h5"
        real_model   = get_path_best_epoch("vgg_5_200_small", full_path=False)
        brokendata_tag = "xzt_broken4"
        realdata_tag   = "xzt"
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        title_of_plot='Small 200 neuron Autoencoder-encoder network performance\nwith manipulated simulations'
        #in the results/plots folder:
        plot_file_name = "vgg_5_200_small_broken4_enc"+extra_name+".pdf" 
        y_lims=(0.7,0.95)
    
    # ----------------------------- Energy regression -----------------------------
    
    elif which_one=="energy_12_enc" or which_one==16:
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_200_small_broken12_enc"+extra_name+".pdf" 
        plot_type = "mre"
        #y_lims=(0.7,0.95)
        
        broken_model = "vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_broken12_epoch48.h5"
        real_model   = get_path_best_epoch("vgg_3_2000_E", full_path=False)
        brokendata_tag = "xzt_broken12"
        realdata_tag   = "xzt"
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        
    elif which_one=="energy_12_unf" or which_one==17:
        brokendata_tag = "xzt_broken12"
        realdata_tag   = "xzt"
        broken_model = "vgg_3-broken12/trained_vgg_3-broken12_supervised_energy_epoch11.h5"
        real_model   = get_path_best_epoch("2000_unf_E", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_200_small_broken12_unf"+extra_name+".pdf" 
        plot_type = "mre"
        #y_lims=(0.7,0.95)
        
        
    elif which_one=="energy_4_2000_unf" or which_one==19:
        brokendata_tag = "xzt_broken4"
        realdata_tag   = "xzt"
        broken_model = "vgg_3-broken4/trained_vgg_3-broken4_supervised_energy_epoch10.h5"
        real_model   = get_path_best_epoch("2000_unf_E", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken4_unf"+extra_name+".pdf" 
        plot_type = "mre"
    elif which_one=="energy_4_2000_enc" or which_one==20:
        brokendata_tag = "xzt_broken4"
        realdata_tag   = "xzt"
        broken_model = "vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_broken4_nodrop_epoch5.h5"
        real_model   = get_path_best_epoch("vgg_3_2000_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken4_enc"+extra_name+".pdf" 
        plot_type = "mre"
        
        
    elif which_one=="energy_13_2000_unf" or which_one==21:
        brokendata_tag = "xzt_broken13"
        realdata_tag   = "xzt"
        broken_model = "vgg_3-broken13/trained_vgg_3-broken13_supervised_energy_epoch19.h5"
        real_model   = get_path_best_epoch("2000_unf_E", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken13_unf"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.02,0.78)

    elif which_one=="energy_13_2000_enc" or which_one==22:
        brokendata_tag = "xzt_broken13"
        realdata_tag   = "xzt"
        broken_model = "vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_broken13_nodrop_epoch9.h5"
        real_model   = get_path_best_epoch("vgg_3_2000_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken13_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.02,0.78)
   
    #Broken 14 (rauschen prop zu E, bis zu 2 kHz plus)
    #Bottleneck scan
    elif which_one=="energy_14_2000_unf" or which_one==24:
        brokendata_tag = "xzt_broken14"
        realdata_tag   = "xzt"
        broken_model = "vgg_3-broken14/trained_vgg_3-broken14_supervised_energy_epoch15.h5"
        real_model   = get_path_best_epoch("2000_unf_E", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken14_unf"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.08,0.68)

    elif which_one=="energy_14_2000_enc" or which_one==25:
        brokendata_tag = "xzt_broken14"
        realdata_tag   = "xzt"
        broken_model = "vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_broken14_nodrop_epoch7.h5"
        real_model   = get_path_best_epoch("vgg_3_2000_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken14_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.08,0.68)
    
    elif which_one=="energy_14_600_pic_enc" or which_one==27:
        brokendata_tag = "xzt_broken14"
        realdata_tag   = "xzt"
        broken_model = "vgg_5_picture/trained_vgg_5_picture_autoencoder_epoch44_supervised_energy_broken14_nodrop_epoch12.h5"
        real_model   = get_path_best_epoch("vgg_5_600_picture_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_picture_broken14_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.08,0.68)
        
    elif which_one=="energy_14_200_dense_enc" or which_one==28:
        brokendata_tag = "xzt_broken14"
        realdata_tag   = "xzt"
        broken_model = "vgg_5_200_dense-new/trained_vgg_5_200_dense-new_autoencoder_epoch101_supervised_energy_broken14_nodrop_epoch45.h5"
        real_model   = get_path_best_epoch("vgg_5_200_dense_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_200_dense_broken14_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.08,0.68)
        
    elif which_one=="energy_14_64_enc" or which_one==29:
        brokendata_tag = "xzt_broken14"
        realdata_tag   = "xzt"
        broken_model = "vgg_5_64/trained_vgg_5_64_autoencoder_epoch78_supervised_energy_broken14_nodrop_epoch49.h5"
        real_model   = get_path_best_epoch("vgg_5_64_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_64_broken14_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.08,0.68)
        
    elif which_one=="energy_14_32_enc" or which_one==30:
        brokendata_tag = "xzt_broken14"
        realdata_tag   = "xzt"
        broken_model = "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch44_supervised_energy_broken14_nodrop_epoch59.h5"
        real_model   = get_path_best_epoch("vgg_5_32_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_32_broken14_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.08,0.68)
        
    elif which_one=="energy_14_200_enc" or which_one==31:
        brokendata_tag = "xzt_broken14"
        realdata_tag   = "xzt"
        broken_model = "vgg_5_200/trained_vgg_5_200_autoencoder_epoch94_supervised_energy_broken14_nodrop_epoch11.h5"
        real_model   = get_path_best_epoch("vgg_5_200_E_nodrop", full_path=False)
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_200_broken14_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.08,0.68)
        
    # ----------------------------- Other tests -----------------------------

    elif which_one=="energy_2_2000_unf" or which_one==32:
        brokendata_tag = "xzt"
        realdata_tag   = "xzt_broken2"
        broken_model = get_path_best_epoch("2000_unf_E", full_path=False)
        real_model   = "vgg_3-noise10/trained_vgg_3-noise10_supervised_energy_epoch12.h5"
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken2_unf"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.21,0.81)
    elif which_one=="energy_2_2000_enc" or which_one==33:
        brokendata_tag = "xzt"
        realdata_tag   = "xzt_broken2"
        broken_model = "vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch5_supervised_energy_nodrop_epoch3.h5"
        real_model   = "vgg_3-noise10/trained_vgg_3-noise10_autoencoder_epoch7_supervised_energy_nodrop_epoch5.h5" #_broken2
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken2_enc"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.21,0.81)
    
    elif which_one=="energy_15_2000_unf" or which_one==34:
        brokendata_tag = "xzt"
        realdata_tag   = "xzt_broken15"
        broken_model = get_path_best_epoch("2000_unf_E", full_path=False)
        real_model   = "vgg_5_2000-broken15/trained_vgg_5_2000-broken15_supervised_energy_epoch12.h5"
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        folder_in_the_plots_path = "broken_study_energy/"
        plot_file_name = "vgg_5_2000_broken15_unf"+extra_name+".pdf" 
        plot_type = "mre"
        y_lims=(0.18,0.55)
    # ----------------------------- Unfreeze stuff -----------------------------
    
    elif which_one=="unfreeze_comp" or which_one==18:
        broken_model = "vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_contE20_broken4_epoch30.h5"
        real_model   = "vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_contE20_epoch30.h5"
        brokendata_tag = "xzt_broken4"
        realdata_tag   = "xzt"
        modelidents, dataset_array = get_procedure(broken_model, real_model, 
                                                   brokendata_tag, realdata_tag)
        #Plot properties: All in the array are plotted in one figure, with own label each
        title_of_plot='Continuation of partially unfrozen network training'
        #in the results/plots folder:
        folder_in_the_plots_path="unfreeze/"
        plot_file_name = "broken4_vgg5_200_contE20"+extra_name+".pdf" 
        #y limits of plot:
        y_lims=(0.7,1.0)
        legend_loc="lower right"
    
    else:
        raise NameError(str(which_one) + " is not known!")
    
    if plot_type=="mre":
        #energy plot
        label_array=["On 'simulations'", "On 'measured' data", "Lower limit on 'measured' data"]
    else:
        label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
        
    if y_lims_override != None:
        y_lims = y_lims_override
        
    modelidents = [modelpath + modelident for modelident in modelidents]
    save_plot_as = plot_path + folder_in_the_plots_path + plot_file_name    
    
    
    return modelidents, dataset_array ,title_of_plot, save_plot_as, y_lims, class_type, plot_type, legend_loc, label_array, color_array


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


def make_evaluation(info_tag, extra_name, y_lims_override, show_the_plot=True):
    """
    Main function:
    Make an evaluation based on the info_tag (Generate+Save or load evaluation data, save plot).
    A plot that shows acc or loss over the mc energy in a histogram, evaluated on different 
    datasets.
    Often, there will be three models plotted: 
        0: On 'simulations'
        1: On 'measured' data
        2: Upper lim
    """
    modelidents, dataset_array, title_of_plot, save_plot_as, y_lims, class_type, plot_type, legend_loc, label_array, color_array = get_info(info_tag, extra_name=extra_name, y_lims_override=y_lims_override)
    
    #make plot of multiple data:
    if plot_type == "acc":
        #For up-down networks:
        #generate or load data automatically:
        #this will be a list of binned evaluations, one for every model
        hist_data_array = make_or_load_files(modelidents, dataset_array, class_type=class_type, bins=bins)
        print_statistics_in_numbers(hist_data_array, plot_type)
        
        y_label_of_plot="Accuracy"
        fig = make_binned_data_plot(hist_data_array, label_array, title_of_plot, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array, legend_loc=legend_loc) 
    
    elif plot_type == "mre":
        #Median relative error for energy regression, seperated for track and shower
        #Data is loaded by the energy evaluation function, which is not
        #fully compatible with this one :-( so additional infos copied from there manually
        hist_data_array=[]
        hist_data_single=[]
        for model_no,model_path in enumerate(modelidents):
            dataset_tag = dataset_array[model_no]
            print("Working on", model_path.split("trained_")[1][:-3], "using dataset", dataset_tag)
            zero_center=True
            energy_bins_2d=np.arange(3,101,1)
            energy_bins_1d=20
            hist_data_2d, energy_mae_plot_data = make_or_load_hist_data(model_path, 
                    dataset_tag, zero_center, energy_bins_2d, energy_bins_1d, samples=None, 
                    include_mae_single=True)
            #only interested in the mae plot data
            hist_data_array.append(energy_mae_plot_data[:2])
            hist_data_single.append(energy_mae_plot_data[2])
            
        print_statistics_in_numbers(hist_data_array, plot_type, hist_data_single=hist_data_single)
        
        y_label_of_plot='Median fractional energy resolution'
        
        #Make the single plot and save without displaying
        fig_single = make_energy_mae_plot_mean_only_single(hist_data_single, label_list=label_array, color_list=color_array, y_lims=y_lims)
        fig_single_save_as=save_plot_as[:-4]+"_single.pdf"
        fig_single.savefig(fig_single_save_as)
        print("Single plot saved to", fig_single_save_as)
        plt.close(fig_single)
        
        fig = make_energy_mae_plot_mean_only(hist_data_array, label_list=label_array, color_list=color_array, y_lims=y_lims)
        
        
    elif plot_type == "mse":
        #Intended for Autoencoders, not been used in a long time...
        y_label_of_plot="Loss"
        fig = make_binned_data_plot(hist_data_array, label_array, title_of_plot, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array, legend_loc=legend_loc) 
    else:
        print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")
    
    
    fig.savefig(save_plot_as)
    print("Plot saved to", save_plot_as)
    
    if show_the_plot == True:
        plt.show(fig)
    else:
        plt.close(fig)
    
    return

def print_statistics_in_numbers(hist_data_array, plot_type, return_line=False, hist_data_single=None):
    """
    Prints the average overall loss of performance, 
    averaged over all bins (not all events).
    For this, three hist_datas are necessary:
    hist_data_array
                [0]:  On simulations (broken on broken)
                [1]:  On measured    (broken on real)
                [2]:  Upper limit    (real on real)
    """
    print("\n----------Statistics of this evaluation-----------------")
    print("\tAveraged over energy bins, not events!")
    
    if plot_type == "acc":
        #hist_data contains [energy, binned_acc] for every model
        on_simulations_data = hist_data_array[0][1]
        on_measured_data    = hist_data_array[1][1]
        upper_limit_data    = hist_data_array[2][1]
        
        dropoff_sim_measured = ( (on_simulations_data - on_measured_data)/on_measured_data ).mean()
        dropoff_upper_limit_measured = ((upper_limit_data - on_measured_data)/on_measured_data ).mean()
        
        print("Acc on Sims:\tOn measured\tUpper lim")
        print(np.mean(on_simulations_data),"\t", np.mean(on_measured_data),"\t", np.mean(upper_limit_data))
        print("\nAverage relative %-acc reduction across all bins: 100 * (x - measured) / measured")
        print("From simulation to measured\tFrom upper lim to measured:")
        print(dropoff_sim_measured*100,"\t",dropoff_upper_limit_measured*100)
        print("--------------------------------------------------------\n")
        header = ("(Sim-Meas)/Meas","(Upperlim-Meas)/Meas")
        line=(dropoff_sim_measured*100, dropoff_upper_limit_measured*100)
        
    elif plot_type=="mre":
        #hist_data_array is for every model the tuple:
        #[energy_mae_plot_data_track, energy_mae_plot_data_shower]
        #each containing [energy, binned mre]      
        #hist_data_single contains for every model the unseperated data tuple: [energy, binned mre]     
        
        on_simulations_data_track = np.array(hist_data_array[0][0][1])
        on_measured_data_track    = np.array(hist_data_array[1][0][1])
        upper_limit_data_track    = np.array(hist_data_array[2][0][1])
        
        on_simulations_data_shower = np.array(hist_data_array[0][1][1])
        on_measured_data_shower    = np.array(hist_data_array[1][1][1])
        upper_limit_data_shower    = np.array(hist_data_array[2][1][1])
        
        on_simulations_data_single = np.array(hist_data_single[0][1])
        on_measured_data_single = np.array(hist_data_single[1][1])
        upper_limit_data_single = np.array(hist_data_single[2][1])
        
        print("First three are MRE, last two are average relative % increase across all bins: -1 * 100 * (x - measured) / measured")
            
        def print_one_table (on_simulations_data, on_measured_data, upper_limit_data, printig_header="Track like events:"):
            dropoff_sim_measured =   (-1*(on_simulations_data  - on_measured_data)/on_measured_data).mean()
            dropoff_upper_limit =    (-1*(upper_limit_data  - on_measured_data )/on_measured_data  ).mean()
            print(printig_header)
            print("On Sims:\tOn measured\tUpper lim\tFrom simulation to measured\tFrom upper lim to measured:")
            print(np.mean(on_simulations_data),"\t", np.mean(on_measured_data),"\t", np.mean(upper_limit_data),"\t", dropoff_sim_measured*100,"\t",dropoff_upper_limit*100) 
            print("--------------------------------------------------------\n")
            
        print_one_table(on_simulations_data_track, on_measured_data_track, upper_limit_data_track, "Track like events:")
        print_one_table(on_simulations_data_shower, on_measured_data_shower, upper_limit_data_shower, "Shower like events:")
        print_one_table(on_simulations_data_single, on_measured_data_single, upper_limit_data_single, "All events:")
        
        header = None
        line=None
        
    else:
        raise NameError("Unknown plottype"+plot_type)
    
    if return_line:
        return header, line
    
    
if __name__ == "__main__":
    params = parse_input()
    which_ones = params["info_tags"]

    if "all" in which_ones:
        show_the_plot = False
        current_tag=0
        while True:
            try:
                make_evaluation(current_tag, extra_name, y_lims_override, show_the_plot)
                current_tag+=1
            except NameError:
                print("Done. Made a total of", current_tag, "plots.")
                break
                
    else:
        show_the_plot = True
        for info_tag in which_ones:
            make_evaluation(info_tag, extra_name, y_lims_override, show_the_plot)
          
        
        
        
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#not supported anymore...

if make_difference_plot == True  or make_difference_plot == "both":
    raise
    #which plots to make diff of; (first - second) / first
    make_diff_of_list=((0,1),(2,1))
    title_list=("Relative loss of accuracy: 'simulations' to 'measured' data",
                "Realtive difference in accuracy: Upper limit to 'measured' data")
    
    if which_broken_study==2:
        which_ones = ("2_unf", "2_enc")
        save_as_list=(plot_path + "vgg_3_broken2_sim_real"+extra_name+".pdf", 
                      plot_path + "vgg_3_broken2_upper_real"+extra_name+".pdf")
        y_lims_list=((-0.02,0.1),(-0.02,0.1))
        
    elif which_broken_study==4:
        which_ones = ("4_unf", "4_enc")
        save_as_list=(plot_path + "vgg_3_broken4_sim_real"+extra_name+".pdf", 
                      plot_path + "vgg_3_broken4_upper_real"+extra_name+".pdf")
        y_lims_list=((-0.02,0.1),(-0.02,0.1))
        
    else:
        raise()
    
    for i in range(len(make_diff_of_list)):
        #label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
        modelidents,dataset_array,title_of_plot,plot_file_name,y_lims = get_info(which_ones[0], y_lims_override=y_lims_override)
        
        modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
        #           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
        for modelident in modelidents:
            modelnames.append(modelident.split("trained_")[1][:-3])
        
        hist_data_array_unf = make_or_load_files(modelnames, dataset_array, modelidents=modelidents, class_type=class_type, bins=bins)
        
        
        modelidents,dataset_array,title_of_plot,plot_file_name,y_lims = get_info(which_ones[1], y_lims_override=y_lims_override)
        
        modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
        #           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
        for modelident in modelidents:
            modelnames.append(modelident.split("trained_")[1][:-3])
            
        hist_data_array_enc = make_or_load_files(modelnames, dataset_array, modelidents=modelidents, class_type=class_type, bins=bins)
        
        
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
        hist_1=np.array(hist_data_array_unf[make_diff_of[0]])
        hist_2=np.array(hist_data_array_unf[make_diff_of[1]])
        diff_hist=[hist_1[0], (hist_1[1]-hist_2[1])/hist_1[1]]
        hist_data_array_diff.append(diff_hist)
        
        hist_1=np.array(hist_data_array_enc[make_diff_of[0]])
        hist_2=np.array(hist_data_array_enc[make_diff_of[1]])
        diff_hist=[hist_1[0], (hist_1[1]-hist_2[1])/hist_1[1]]
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
