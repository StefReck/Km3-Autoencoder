# -*- coding: utf-8 -*-
"""
Plot of AE loss vs Encoder acc of several networks.
"""
import numpy as np
import matplotlib.pyplot as plt

from plot_statistics import make_data_from_files, get_last_prl_epochs

test_files_ae = ["models/vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt",
                 "models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_test.txt",
                 
                 "models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt",
                 "models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_test.txt",
                 
                 "models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt",
                 "models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"]

test_files_prl = ["models/vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_up_down_new_test.txt",
                  "models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_supervised_parallel_up_down_test.txt",
                  
                  "models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_up_down_test.txt",
                  "models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_supervised_parallel_up_down_test.txt",
                  
                  "models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_up_down_test.txt",
                  "models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_up_down_test.txt"]

how_many_epochs_each_to_train_list = [ [10,]*1+[2,]*5+[1,]*194,
                                       [10,]*1+[2,]*5+[1,]*194, 
                                       [10,]*1+[2,]*5+[1,]*194,
                                       [10,]*1+[2,]*5+[1,]*194,
                                       [10,]*1+[2,]*5+[1,]*194, 
                                       [10,]*1+[2,]*5+[1,]*194,]

label_array_overwrite = ["Autoencoder 600 (picture)",
                         "Autoencoder 600 (filters)",
                         "Autoencoder 200",
                         "Autoencoder 200 (dense)",
                         "Autoencoder 64",
                         r"Autoencoder 32 ($\epsilon = 10^{-1}$)",]


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
    fig, ax = plt.subplots(figsize=(10,7))
    for i,model_loss_acc in enumerate(loss_acc_list):
        ax.plot(model_loss_acc[0], model_loss_acc[1], "o", ms=5, label=labels[i])
    ax.set_xlabel("Autoencoder loss")
    ax.set_ylabel("Encoder accuracy")
    ax.grid()
    ax.legend(loc="lower left")
    fig.suptitle("Autoencoder loss and encoder accuracy for different autoencoder models")
    return fig, ax

if len(label_array_overwrite)==len(default_label_array_ae):
    label_array = label_array_overwrite
else:
    label_array = default_label_array_ae

fig, ax = make_plot(loss_acc_list, label_array)
plt.show(fig)




    
