# -*- coding: utf-8 -*-
"""
Plot of AE loss vs Encoder acc of several networks.
"""
import numpy as np
import matplotlib.pyplot as plt

from plot_statistics import make_data_from_files, get_last_prl_epochs

test_files_ae = ["vgg_5_200/trained_vgg_5_200_autoencoder_test.txt",]
test_files_prl = ["vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_up_down_test.txt",]

how_many_epochs_each_to_train_list = [ [10,]*1+[2,]*5+[1,]*194, ]*len(test_files_ae)

data_for_plots_ae, ylabel_list_ae = make_data_from_files(test_files_ae)
data_for_plots_prl, ylabel_list_prl = make_data_from_files(test_files_prl)
#data_for_plots:
# [  [Test_epoch, Test_ydata, Train_epoch, Train_ydata],...    ]
#                   for every test file, ....

loss_acc_list = []
for ae_number in range(len(test_files_ae)):
    data_ae = data_for_plots_ae[ae_number]
    data_prl =data_for_plots_prl[ae_number]
    #[Test_epoch, Test_ydata, Train_epoch, Train_ydata]
    how_many_epochs_each_to_train = how_many_epochs_each_to_train_list[ae_number]
    
    data_parallel_test, data_parallel_train = get_last_prl_epochs(data_ae, data_prl, how_many_epochs_each_to_train)
    #[epoch, ydata]
    
    loss_acc = [[],[]]
    for autoencoder_epoch in data_ae[0]:
        #ae loss of specific ae epoch
        ae_loss = data_ae[1][autoencoder_epoch]
        #encoder accuracy of a specific ae epoch
        enc_acc = data_parallel_test[1][np.where(data_parallel_test[0]==autoencoder_epoch)]
    
        if len(enc_acc) != 0:
            loss_acc[0].append(ae_loss)
            loss_acc[1].append(enc_acc)
    loss_acc_list.append(loss_acc)


for model_loss_acc in loss_acc_list:
    plt.plot(model_loss_acc[0], model_loss_acc[1])
plt.show()




    