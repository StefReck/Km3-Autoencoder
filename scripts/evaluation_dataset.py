# -*- coding: utf-8 -*-

"""
Evaluate model performance after training. 
This is for comparison of supervised accuracy on different datasets.
Especially for the plots for the broken data comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from util.evaluation_utilities import make_or_load_files, make_energy_to_accuracy_plot_comp_data, make_energy_to_loss_plot_comp_data



#extra string to be included in file names
extra_name=""
#number of bins; default is 97; backward compatibility with 98 bins
bins=32

#Standard, plot acc vs energy plots of these:
which_ones=("1_unf","1_enc")
#If not None: Change the y range of all plots to this one (to make unit looks)
y_lims_override = None
#Override default location of legend ("best")
legend_loc="center right"
#instead of plotting acc vs. energy, one can also make a compare plot, 
#which shows the difference #between "on simulations" and "on measured data"
#then, the number of the broken mode has to be given
#can be True, False or "both"
make_difference_plot=False
which_broken_study=4




extra_name="_"+ str(bins)+"_bins" + extra_name

def get_info(which_one, extra_name="", y_lims_override=None):
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
    
    else:
        print(which_one, "is not known!")
        raise(TypeError)
        
    if y_lims_override != None:
        y_lims = y_lims_override
        
    return modelidents,dataset_array,title_of_plot,plot_file_name,y_lims

#label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    

label_array=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
#Overwrite default color palette. Leave empty for auto
color_array=["orange", "blue", "navy"]
#loss, acc, None
plot_type = "acc"
#Info about model
class_type = (2, 'up_down')


modelpath = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
plot_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"

if make_difference_plot == False or make_difference_plot == "both":
    for which_one in which_ones:
        
        modelidents,dataset_array,title_of_plot,plot_file_name,y_lims = get_info(which_one, extra_name=extra_name, y_lims_override=y_lims_override)
        
        modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
        #           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
        for modelident in modelidents:
            modelnames.append(modelident.split("trained_")[1][:-3])
            
        save_plot_as = plot_path + plot_file_name
        
        #generate or load data automatically:
        hist_data_array = make_or_load_files(modelnames, dataset_array, modelidents=modelidents, modelpath=modelpath, class_type=class_type, bins=bins)
        #make plot of multiple data:
        if plot_type == "acc":
            y_label_of_plot="Accuracy"
            make_energy_to_accuracy_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, y_lims=y_lims, color_array=color_array, legend_loc=legend_loc) 
        elif plot_type == "loss":
            y_label_of_plot="Loss"
            make_energy_to_loss_plot_comp_data(hist_data_array, label_array, title_of_plot, filepath=save_plot_as, y_label=y_label_of_plot, color_array=color_array) 
        elif plot_type == None:
            print("plot_type==None: Not generating plots")
        else:
            print("Plot type", plot_type, "not supported. Not generating plots, but hist_data is still saved.")
        
        print("Plot saved to", save_plot_as)
            
    
if make_difference_plot == True or make_difference_plot == "both":
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
        
        hist_data_array_unf = make_or_load_files(modelnames, dataset_array, modelidents=modelidents, modelpath=modelpath, class_type=class_type, bins=bins)
        
        
        modelidents,dataset_array,title_of_plot,plot_file_name,y_lims = get_info(which_ones[1], y_lims_override=y_lims_override)
        
        modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
        #           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
        for modelident in modelidents:
            modelnames.append(modelident.split("trained_")[1][:-3])
            
        hist_data_array_enc = make_or_load_files(modelnames, dataset_array, modelidents=modelidents, modelpath=modelpath, class_type=class_type, bins=bins)
        
        
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
