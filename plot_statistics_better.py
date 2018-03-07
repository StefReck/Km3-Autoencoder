# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_statistics import make_data_from_files, make_plot_same_y
matplotlib.rcParams.update({'font.size': 14})

"""
Make a plot of multiple models, each identified with its test log file, with
lots of different options.
This is intended for plotting models with the same y axis data (loss OR acc).
"""

xlabel="Epoch"
title="Loss of autoencoders with a varying number of convolutional layers"
figsize = (9,6)
#Override default labels (names of the models); must be one for every test file, otherwise default
labels_override=["12 layers CW", "12 layers","14 layers", "16 layers", "20 layers"]
#legend location for the labels and the test/train box
legend_locations=(1, "upper left")
#Override xtick locations; None for automatic
xticks=None
# override line colors; must be one color for every test file, otherwise automatic
colors=[] # = automatic
#Name of file to save the numpy array with the plot data to; None will skip saving
dump_to_file=None


#Generate dummy data instead of actually reading in the files
debug=1



#Test Files to make plots from; Format: vgg_3/trained_vgg_3_autoencoder_test.txt
#Autoencoders:
test_files = ["vgg_3/trained_vgg_3_autoencoder_test.txt",
              #"vgg_3_reg-e9/trained_vgg_3_reg-e9_autoencoder_test.txt",
              #"vgg_3-eps4/trained_vgg_3-eps4_autoencoder_test.txt",
              #"vgg_3_dropout/trained_vgg_3_dropout_autoencoder_test.txt",
              #"vgg_3_max/trained_vgg_3_max_autoencoder_test.txt",
              #"vgg_3_stride/trained_vgg_3_stride_autoencoder_test.txt",
              #"vgg_3_stride_noRelu/trained_vgg_3_stride_noRelu_autoencoder_test.txt",
              "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",]
              #"vgg_3_small/trained_vgg_3_small_autoencoder_test.txt",
              #"vgg_3_verysmall/trained_vgg_3_verysmall_autoencoder_test.txt",]

#Unfrozen
xtest_files = ["vgg_3/trained_vgg_3_supervised_up_down_test.txt",
              "vgg_3_dropout/trained_vgg_3_dropout_supervised_up_down_test.txt",
              "vgg_3_max/trained_vgg_3_max_supervised_up_down_test.txt",
              "vgg_3_stride/trained_vgg_3_stride_supervised_up_down_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_supervised_up_down_test.txt",]

#Encoders Epoch 10
xtest_files = ["vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_dropout/trained_vgg_3_dropout_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_max/trained_vgg_3_max_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_stride/trained_vgg_3_stride_autoencoder_epoch10_supervised_up_down_test.txt", ]

#sgdlr01 encoders
xtest_files = ["vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch2_supervised_up_down_test.txt",
              "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch5_supervised_up_down_test.txt",
              "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch10_supervised_up_down_test.txt",]

#Enocders vgg_3_eps AE E10
xtest_files = ["vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_triplebatchnorm_e1_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_unfbatchnorm_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch80_supervised_up_down_unfbatchnorm_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_unfbatchnorm_no_addBN_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_batchnorm_e1_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch11_supervised_up_down_test.txt",]

#vgg3eps AE E10 Encoders: finaler test von vgg_3
xtest_files = [ "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_zero_center_and_norm_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_BN_noBN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_norm_noBN_noDrop_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_downNorm_BN_noDrop_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_BN_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_noBN_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_zero_center_test.txt",]

#only
xtest_files = ["vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_up_down_new_test.txt",]

#vgg4 autoencoders variational depth
xtest_files = [#"vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
              "vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
              "vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
              "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
              "vgg_4_15c/trained_vgg_4_15c_autoencoder_test.txt",
              "vgg_4_30c/trained_vgg_4_30c_autoencoder_test.txt",]

xtitle="Loss of autoencoders with a varying number of convolutional layers"
xticks=[0,5,10,15,20,25,30]
xlabels_override=["12 layers","14 layers", "16 layers", "20 layers", "30 layers", "60 layers"]


#vgg4 autoencoders 10c tests
test_files = [#"vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
              #"vgg_4_ConvAfterPool/trained_vgg_4_ConvAfterPool_autoencoder_test.txt",
              #"vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
              #"vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
              #"vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
              "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
              "vgg_4_10c_smallkernel/trained_vgg_4_10c_smallkernel_autoencoder_test.txt",
              "vgg_4_10c_triple/trained_vgg_4_10c_triple_autoencoder_test.txt",
              "vgg_4_10c_triple_same_structure/trained_vgg_4_10c_triple_same_structure_autoencoder_test.txt",]
              #"vgg_4_7c_less_filters/trained_vgg_4_7c_less_filters_autoencoder_test.txt"]
title="Loss of autoencoders with 20 convolutional layers"
labels_override=["Standard", "Small kernel", "Triple structure", "Triple structure variation"]

#vgg_5 smaller bottleneck
xtest_files = ["vgg_5_channel/trained_vgg_5_channel_autoencoder_test.txt",
              "vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt",
              "vgg_3/trained_vgg_3_autoencoder_test.txt",
              "vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_test.txt",]


#vgg 3 parallel training
"""
test_files = ["vgg_3/trained_vgg_3_autoencoder_supervised_parallel_up_down_test.txt",]
title="Accuracy during parallel supervised training"
labels_override=[r"Autoencoder with $\epsilon = 10^{-1}$"]
xticks = np.arange(0,51,5)
legend_locations=( "lower right", "upper left") #(labels, Test/Train)
colors=["orange",]
"""



for i,test_file in enumerate(test_files):
    test_files[i] = "models/"+test_file

if debug==False:
    #Generate data from files (can also save it)
    # [Test_epoch, Test_ydata, Train_epoch, Train_ydata], ylabels
    data_for_plots, ylabel_list, default_label_array = make_data_from_files(test_files, dump_to_file=dump_to_file)
else:
    # [Test_epoch, Test_ydata, Train_epoch, Train_ydata]
    data_for_plots=[[np.linspace(0,9,10), np.linspace(0,9,10)*0.1, np.linspace(0,9,10)*0.1 + 1, np.linspace(0,9,10) ], ]
    data_for_plots=np.array(data_for_plots*2)
    ylabel_list=("test","test")


fig = make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, 
                legend_locations, labels_override, colors, xticks, figsize=figsize)
plt.show(fig)




