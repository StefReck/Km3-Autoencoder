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

def which_plot(do_you_want):
    #The folder the plots might be saved to
    base_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/loss_vs_ydata/"
    if do_you_want == "acc": 
        #tags of the models to plot, as defined in saved_setups_for_plots
        tags=["vgg_3",
              "vgg_5_600_morefilter",
              "vgg_5_600_picture",
              "vgg_5_200",
              "vgg_5_200_dense",
              "vgg_5_64",
              "vgg_5_32-eps01",
              #"vgg_5_600-ihlr",
              ]
        
        #Stuff for the plot
        label_list = ["2000",
                      "600 (morefilter)", 
                      "600 (picture)", 
                      "200",
                      "200 (dense)", "64",
                      r"32 ($\epsilon = 10^{-1}$)",
                      #"Autoencoder 600 (picture) high lr",
                      ]
        xlabel, ylabel = "Autoencoder loss", "Encoder accuracy"
        title = "Autoencoder loss and encoder accuracy"
        save_as = "vgg_5_acc.pdf"
        
    elif do_you_want=="loss":
        raise NameError("Nothing is here yet...")
        
    elif do_you_want=="test":
        tags=["vgg_5_600_picture",]
        label_list = ["Autoencoder 600 (picture)",]
        xlabel, ylabel = "Autoencoder loss", "Encoder accuracy"
        title = "Autoencoder loss and encoder accuracy for different autoencoder models"
        save_as = None
    
    if save_as is not None: save_as=base_path+save_as 
    return tags, label_list,xlabel,ylabel,title,save_as



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


def make_plot(loss_ydata_list, labels, xlabel, ylabel, title):
    figsize, font_size = get_plot_statistics_plot_size("two_in_one_line")
    plt.rcParams.update({'font.size': font_size})
    fig, ax=plt.subplots(figsize=figsize)
    
    for i,model_loss_ydata in enumerate(loss_ydata_list):
        ax.plot(model_loss_ydata[0], model_loss_ydata[1], "o", ms=2, label=labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(loc="lower left")
    fig.suptitle(title)
    #plt.gcf().subplots_adjust(left=0., right=1.05, bottom=0, top=1)
    return fig


def make_the_plot(tags, label_list, xlabel, ylabel, title, save_as):
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
    
    fig = make_plot(loss_ydata_list, label_array ,xlabel, ylabel, title)
    
    if save_as is not None:
        fig.savefig(save_as)
        print("Plot saved to", save_as)
    
    return fig


fig = make_the_plot(*which_plot(i_want))
plt.show(fig)


    
