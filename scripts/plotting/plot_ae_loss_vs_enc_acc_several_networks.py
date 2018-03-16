# -*- coding: utf-8 -*-
"""
Plot of AE loss vs Encoder acc of several networks.
"""
import numpy as np
import matplotlib.pyplot as plt
#TODO Bugfix, because picture parallel does not look here like it does with the plot_parallel script
from plot_statistics import make_data_from_files, get_last_prl_epochs, get_default_labels

test_files_ae = ["models/vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt",
                 "vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_test.txt",
                 
                 "models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt",
                 "models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_test.txt",
                 
                 "vgg_5_64/trained_vgg_5_64_autoencoder_test.txt",
                 "vgg_5_32/trained_vgg_5_32_autoencoder_test.txt"]

test_files_prl = ["models/vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_up_down_new_test.txt",
                  "vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_supervised_parallel_up_down_test.txt",
                  
                  "models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_up_down_test.txt",
                  "models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_supervised_parallel_up_down_test.txt",
                  
                  "vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_up_down_test.txt",
                  "vgg_5_32/trained_vgg_5_32_autoencoder_supervised_parallel_up_down_test.txt"]

how_many_epochs_each_to_train_list = [ [10,]*1+[2,]*5+[1,]*194,
                                       [10,]*1+[2,]*5+[1,]*194, 
                                       [10,]*1+[2,]*5+[1,]*194,
                                       [10,]*1+[2,]*5+[1,]*194,
                                       [10,]*1+[2,]*5+[1,]*194, 
                                       [10,]*1+[2,]*5+[1,]*194,]


data_for_plots_ae, ylabel_list_ae, default_label_array_ae = make_data_from_files(test_files_ae)
data_for_plots_prl, ylabel_list_prl, default_label_array_prl = make_data_from_files(test_files_prl)
#data_for_plots:
# [  [Test_epoch, Test_ydata, Train_epoch, Train_ydata],...    ]
#                   for every test file, ....

loss_acc_list = []
for ae_number in range(len(test_files_ae)):
    data_ae = data_for_plots_ae[ae_number]
    data_prl =data_for_plots_prl[ae_number]
    #[Test_epoch, Test_ydata, Train_epoch, Train_ydata]
    how_many_epochs_each_to_train = np.array(how_many_epochs_each_to_train_list[ae_number])
    
    data_parallel_test, data_parallel_train = get_last_prl_epochs(data_ae, data_prl, how_many_epochs_each_to_train)
    #[epoch, ydata]
    
    loss_acc = [[],[]]
    for autoencoder_epoch in data_ae[0]:
        #ae loss of specific ae epoch
        ae_loss = data_ae[1][autoencoder_epoch-1]
        #encoder accuracy of a specific ae epoch
        enc_acc = data_parallel_test[1][np.where(data_parallel_test[0]==autoencoder_epoch)]
    
        if len(enc_acc) != 0:
            loss_acc[0].append(ae_loss)
            loss_acc[1].append(enc_acc)
    loss_acc_list.append(loss_acc)

def make_plot(loss_acc_list, labels):
    fig, ax = plt.subplots(figsize=(8,8))
    for i,model_loss_acc in enumerate(loss_acc_list):
        ax.plot(model_loss_acc[0], model_loss_acc[1], "o", label=labels[i])
    ax.set_xlabel("Autoencoder loss")
    ax.set_ylabel("Encoder accuracy")
    ax.grid()
    ax.legend()
    return fig, ax

fig, ax = make_plot(loss_acc_list, default_label_array_ae)
plt.show(fig)




    