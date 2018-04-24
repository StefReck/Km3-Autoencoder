# -*- coding: utf-8 -*-

"""
Evaluate model performance after training. 
This is for comparison of supervised accuracy on different datasets.
Especially for the plots for the broken data comparison.
"""

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Evaluate model performance after training. This is for comparison of supervised accuracy on different datasets. Especially for the plots for the broken data comparison.')
    parser.add_argument('info_tags', nargs="+", type=str, help='Names of identifiers for a saved setup.')

    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
which_ones = params["info_tags"]


import numpy as np
import matplotlib.pyplot as plt

from util.evaluation_utilities import make_or_load_files, make_binned_data_plot, make_energy_mae_plot


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
#TODO Rework
make_difference_plot=False
which_broken_study=4



#Add the number of bins to the name of the plot file (usually 32)
extra_name="_"+ str(bins)+"_bins" + extra_name

def get_info(which_one, extra_name="", y_lims_override=None):
    """
    Saved setups of plots
    """
    #This will be added before all modelidents
    modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    #Default class type the evaluation is done for. None for autoencoders.
    class_type = (2, 'up_down')
    #mse, acc, mre
    plot_type = "acc"
    #Default location of legend ("best")
    legend_loc="best"
    
    if which_one=="1_unf":
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
        
    
    elif which_one=="1_enc":
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
        legend_loc="lower rigth"
    
    elif which_one=="2_unf":
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
        y_lims=(0.73,0.96)
        
    elif which_one=="2_enc":
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
        y_lims=(0.68,0.92)
    
    elif which_one=="4_unf":
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
    
    elif which_one=="4_enc":
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
        
    elif which_one=="4_pic_enc":
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
    
    elif which_one=="4_200_enc":
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
    
    elif which_one=="4_64_enc":
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
    
    elif which_one=="4flip_unf":
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
    elif which_one=="4flip_enc":
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
        
    elif which_one=="5_enc":
        pass
    elif which_one=="5_unf":
        pass
    
    else:
        print(which_one, "is not known!")
        raise(TypeError)
        
    if y_lims_override != None:
        y_lims = y_lims_override
        
    modelidents = [modelpath + modelident for modelident in modelidents]
        
    return modelidents, dataset_array ,title_of_plot, plot_file_name, y_lims, class_type, plot_type, legend_loc


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    

label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
#Overwrite default color palette. Leave empty for auto
color_array=["orange", "blue", "navy"]



plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/broken_study/"


def make_evaluation(info_tag, extra_name, y_lims_override):
    """
    Make an evaluation based on the info_tag (Generate+Save or load evaluation data, save plot).
    A plot that shows acc or loss over the mc energy in a histogram, evaluated on different 
    datasets.
    Often, there will be three models plotted: 
        0: On 'simulations'
        1: On 'measured' data
        2: Upper limit on 'measured' data
    """
    modelidents, dataset_array, title_of_plot, plot_file_name, y_lims, class_type, plot_type, legend_loc = get_info(info_tag, extra_name=extra_name, y_lims_override=y_lims_override)                
    save_plot_as = plot_path + plot_file_name
    
    #generate or load data automatically:
    #this will be a list of binned evaluations, one for every model
    hist_data_array = make_or_load_files(modelidents, dataset_array, class_type=class_type, bins=bins)
    print_statistics_in_numbers(hist_data_array, plot_type)
    
    #make plot of multiple data:
    if plot_type == "acc":
        y_label_of_plot="Accuracy"
        make_binned_data_plot(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array, legend_loc=legend_loc) 
    
    elif plot_type == "mse":
        y_label_of_plot="Loss"
        make_binned_data_plot(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array, legend_loc=legend_loc) 
    
    elif plot_type == "mre":
        #Median relative error
        y_label_of_plot='Median fractional energy resolution'
        fig = make_energy_mae_plot(hist_data_array, label_list=label_array)
        fig.savefig(save_plot_as)
        plt.close(fig)
        
    elif plot_type == None:
        print("plot_type==None: Not generating plots")
    else:
        print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")
    
    print("Plot saved to", save_plot_as)
    return

def print_statistics_in_numbers(hist_data_array, plot_type):
    """
    Prints the average overall loss of performance, 
    middled over all bins (not all events).
    """
    if plot_type == "acc":
        #hist_data contains [energy, binned_acc] for every model
        on_simulations_data = hist_data_array[0][1]
        on_measured_data    = hist_data_array[1][1]
        upper_limit_data    = hist_data_array[2][1]
        
        dropoff_sim_measured = np.abs(on_simulations_data - on_measured_data).mean()
        dropoff_upper_limit_measured = np.abs(upper_limit_data - on_measured_data).mean()
        
        print("Average acc reduction across all bins:")
        print("From simulation to measured:", dropoff_sim_measured)
        print("From upper lim to measured:", dropoff_upper_limit_measured)
    elif plot_type=="mre":
        pass
    
    else:
        raise NameError("Unknown plottype"+plot_type)
    

if make_difference_plot == False or make_difference_plot == "both":
    for info_tag in which_ones:
        make_evaluation(info_tag, extra_name, y_lims_override)
      
        
if make_difference_plot == True  or make_difference_plot == "both":
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
