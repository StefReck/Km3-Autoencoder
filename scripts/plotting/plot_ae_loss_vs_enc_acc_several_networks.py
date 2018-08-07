# -*- coding: utf-8 -*-
"""
Plot of AE loss vs Encoder acc of several networks.
This is used to show that all the networks get worse at a certain AE loss.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/woody/capn/mppi013h/Km3-Autoencoder/scripts/')

from plot_statistics import make_data_from_files, get_last_prl_epochs
from util.saved_setups_for_plot_statistics import get_props_for_plot_parallel, get_how_many_epochs_each_to_train
from util.saved_setups_for_plot_statistics import get_plot_statistics_plot_size

#Which plot do
i_want = "acc"
#loss acc test

def which_plot(do_you_want):
    #The folder the plots might be saved to
    base_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/loss_vs_ydata/"
    if do_you_want == "acc": 
        #tags of the models to plot, as defined in 
        #saved_setups_for_plots: get_props_for_plots_parallel
        tags=["vgg_3",
              "vgg_5_600_morefilter",
              "vgg_5_600_picture",
              #"vgg_5_600-ihlr",
              "vgg_5_200",
              "vgg_5_200_large",
              #"vgg_5_200_dense",
              "vgg_5_64",
              "vgg_5_32-eps01",
              ]
        #Stuff for the plot
        label_list = ["model-1920",
                      "600-75 filters", 
                      "600-50 filters", 
                      #"600 (picture) high lr",
                      "200-conv",
                      "200-wide",
                      #"200 (dense)", 
                      "64",
                      "32",
                      ]
        xlabel, ylabel = "Autoencoder loss", "Encoder+dense accuracy"
        title = ""#"Autoencoder loss and encoder accuracy"
        save_as = "vgg_5_acc.pdf"
        #x and y lims
        limits=[[0.064, 0.074],[0.778,0.858]]
        smooth_values=1
        
    elif do_you_want=="loss":
        tags=["vgg_3_energy",
              "vgg_5_600_morefilter_energy",
              "vgg_5_600_picture_energy",
              "vgg_5_200_energy",
              "vgg_5_200_dense_energy",
              "vgg_5_64_energy_nodrop",
              "vgg_5_32-eps01_energy_nodrop",
              ]
        #Stuff for the plot
        label_list = ["model-1920",
                      "600-75 filters", 
                      "600-50 filters", 
                      "200-conv",
                      "200-dense", 
                      "64",
                      "32",
                      ]
        xlabel, ylabel = "Autoencoder loss", "Encoder+dense loss"
        title = ""#"Autoencoder loss and encoder accuracy"
        save_as = "vgg_5_ergy.pdf"
        #x and y lims
        limits=[[0.064, 0.074], [7.2,5.8]]
        smooth_values=3
        
    elif do_you_want=="test":
        tags=["vgg_5_600_picture",]
        label_list = ["Autoencoder 600 (picture)",]
        xlabel, ylabel = "Autoencoder loss", "Encoder accuracy"
        title = "Autoencoder loss and encoder accuracy for different autoencoder models"
        save_as = None
        smooth_values=0
    
    if save_as is not None: save_as=base_path+save_as 
    return tags, label_list,xlabel,ylabel,title,save_as, limits, smooth_values



def combine_ae_and_parallel(data_ae, data_prl, epoch_schedule):
    """
    Takes two list with: AE losses over epoch; and spvsd ydata over epoch,
    and gives out AE loss over ydata (each on the same epoch)
    Output:   list len 2 containing: list of AE loss, list of Prl ydata
    """
    #data_XXX contains: [Test_epoch, Test_ydata, Train_epoch, Train_ydata]
    how_many_epochs_each_to_train = get_how_many_epochs_each_to_train(epoch_schedule)
    
    data_parallel_test, data_parallel_train = get_last_prl_epochs(data_ae, data_prl, how_many_epochs_each_to_train)
    #[epoch, ydata]
    
    loss_ydata = [[],[]]
    for autoencoder_epoch in data_ae[0]:
        #ae loss of specific ae epoch, the loss of AE E1 is at position [0] etc.
        ae_loss = data_ae[1][autoencoder_epoch-1]
        #encoder accuracy of a specific ae epoch
        enc_ydata = data_parallel_test[1][np.where(data_parallel_test[0]==autoencoder_epoch)]
    
        if len(enc_ydata) != 0:
            loss_ydata[0].append(ae_loss)
            loss_ydata[1].append(enc_ydata)
    return loss_ydata

def smooth(values, span=1):
    #average over span*2 + 1 values
    smoothed_values=[]
    for value_index in range( span, len(values)-span ):
        smoothed_values.append(np.mean(values[(value_index-span):(value_index+span+1)]))
    return np.array(smoothed_values)

def make_plot(loss_ydata_list, labels, xlabel, ylabel, title, limits, smooth_values=0):
    figsize, font_size = get_plot_statistics_plot_size("two_in_one_line")
    plt.rcParams.update({'font.size': font_size})
    fig, ax=plt.subplots(figsize=figsize)
    
    for i,model_loss_ydata in enumerate(loss_ydata_list):
        ae_loss, enc_loss = model_loss_ydata[0], model_loss_ydata[1]
        if smooth_values>0:
            temp_plot = ax.plot(ae_loss, enc_loss, "o", ms=1.5,)
            ax.plot(smooth(ae_loss,smooth_values) , smooth(enc_loss,smooth_values), "-", lw=2, color=temp_plot[0].get_color(), label=labels[i])
        else:
            ax.plot(ae_loss, enc_loss, "o-", ms=3, label=labels[i])
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(loc="lower left", fontsize = 10)
    fig.suptitle(title)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    #plt.gcf().subplots_adjust(left=0., right=1.05, bottom=0, top=1)
    return fig


def make_the_plot(tags, label_list, xlabel, ylabel, title, save_as, limits, smooth_values):
    """
    Main function. Returns the plot.
    """
    #Get the names of the _test.txt files (AE and parallel) and the epoch schedule
    #of the models identified with the tags
    test_files_ae, test_files_prl, epoch_schedule_list = [],[],[]
    for tag in tags:
        #info bundle: [ae_model, prl_model], title, labels_override, save_as, epoch_schedule
        info_bundle = get_props_for_plot_parallel(tag)
        ae_file = info_bundle[0][0]
        prl_file= info_bundle[0][1]
        epoch_schedule = info_bundle[4]
        
        test_files_ae.append(ae_file)
        test_files_prl.append(prl_file)
        epoch_schedule_list.append(epoch_schedule)
    
    
    #Read out the info from all the files listed above
    data_for_plots_ae, ylabel_list_ae, default_label_array_ae = make_data_from_files(test_files_ae)
    data_for_plots_prl, ylabel_list_prl, default_label_array_prl = make_data_from_files(test_files_prl)
    #data_for_plots:
    # [  [Test_epoch, Test_ydata, Train_epoch, Train_ydata],...    ]
    #                   for every test file, ....
    
    loss_ydata_list = []
    for ae_number in range(len(test_files_ae)):
        #Go thorugh all the AE-prl doubletts and combine them
        data_ae        = data_for_plots_ae[ae_number]
        data_prl       = data_for_plots_prl[ae_number]
        epoch_schedule = epoch_schedule_list[ae_number]
        
        loss_ydata = combine_ae_and_parallel(data_ae, data_prl, epoch_schedule)
        loss_ydata_list.append(loss_ydata)
    
    
    
    #Define labels for the plot
    if len(label_list)==len(default_label_array_ae):
        label_array = label_list
    else:
        label_array = default_label_array_ae
    
    fig = make_plot(loss_ydata_list, label_array ,xlabel, ylabel, title, limits, smooth_values=smooth_values)
    
    if save_as is not None:
        fig.savefig(save_as)
        print("Plot saved to", save_as)
    
    return fig


fig = make_the_plot(*which_plot(i_want))
plt.show(fig)


    
