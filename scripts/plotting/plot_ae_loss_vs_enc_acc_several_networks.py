# -*- coding: utf-8 -*-
"""
Plot of AE loss vs Encoder acc of several networks.
This is used to show that all the networks get worse at a certain AE loss.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from plot_statistics import make_data_from_files, get_last_prl_epochs
from util.saved_setups_for_plot_statistics import get_props_for_plot_parallel, get_how_many_epochs_each_to_train

#tags of the models to plot, as defined in saved_setups_for_plots
tags=["vgg_5_600_picture","vgg_5_200",
      "vgg_5_200_dense","vgg_5_64",
      "vgg_5_32-eps01"]

#labels for the plot
label_array_overwrite = ["Autoencoder 600 (picture)", "Autoencoder 200",
                         "Autoencoder 200 (dense)", "Autoencoder 64",
                         r"Autoencoder 32 ($\epsilon = 10^{-1}$)",]


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


def make_plot(loss_ydata_list, labels):
    fig, ax = plt.subplots(figsize=(10,7))
    for i,model_loss_ydata in enumerate(loss_ydata_list):
        ax.plot(model_loss_ydata[0], model_loss_ydata[1], "o", ms=5, label=labels[i])
    ax.set_xlabel("Autoencoder loss")
    ax.set_ylabel("Encoder accuracy")
    ax.grid()
    ax.legend(loc="lower left")
    fig.suptitle("Autoencoder loss and encoder accuracy for different autoencoder models")
    return fig



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
if len(label_array_overwrite)==len(default_label_array_ae):
    label_array = label_array_overwrite
else:
    label_array = default_label_array_ae

fig = make_plot(loss_ydata_list, label_array)
plt.show(fig)




    
