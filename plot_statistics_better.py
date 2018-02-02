# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

matplotlib.rcParams.update({'font.size': 14})

xlabel="Epoch"
title="Loss of autoencoders with a varying number of convolutional layers"
labels_override=["12 layers CW", "12 layers","14 layers", "16 layers", "20 layers"]
legend_locations=( 1, "upper left") #(labels, Test/Train)
xticks=None # = automatic
colors=[] # = automatic

#Name of file to save the numpy array with the plot data to; None will skip saving
dump_to_file=None

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
test_files = ["vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_epoch30_supervised_up_down_test.txt",]

#vgg4 autoencoders variational depth
xtest_files = ["vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
              "vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
              "vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
              "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
              "vgg_4_15c/trained_vgg_4_15c_autoencoder_test.txt",
              "vgg_4_30c/trained_vgg_4_30c_autoencoder_test.txt",]

xtitle="Loss of autoencoders with a varying number of convolutional layers"
xlabels_override=["12 layers CW", "12 layers","14 layers", "16 layers", "20 layers", "30 layers"]


#vgg4 autoencoders
xtest_files = ["vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
              #"vgg_4_ConvAfterPool/trained_vgg_4_ConvAfterPool_autoencoder_test.txt",
              #"vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
              #"vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
              #"vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
              "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
              "vgg_4_10c_smallkernel/trained_vgg_4_10c_smallkernel_autoencoder_test.txt",
              "vgg_4_10c_triple/trained_vgg_4_10c_triple_autoencoder_test.txt",
              "vgg_4_10c_triple_same_structure/trained_vgg_4_10c_triple_same_structure_autoencoder_test.txt",
              "vgg_4_7c_less_filters/trained_vgg_4_7c_less_filters_autoencoder_test.txt"]
#vgg_5 smaller bottleneck
test_files = ["vgg_5_channel/trained_vgg_5_channel_autoencoder_test.txt",
              "vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt",
              "vgg_3/trained_vgg_3_autoencoder_test.txt",]


#vgg 3 parallel training
"""
test_files = ["vgg_3/trained_vgg_3_autoencoder_supervised_parallel_up_down_test.txt",]
title="Accuracy during parallel supervised training"
labels_override=[r"Autoencoder with $\epsilon = 10^{-1}$"]
xticks = np.arange(0,51,5)
legend_locations=( "lower right", "upper left") #(labels, Test/Train)
colors=["orange",]
"""

modelnames=[] #e.g. vgg_3_autoencoder
for modelident in test_files:
    modelnames.append(modelident.split("trained_")[1][:-9])
    
    
def make_dicts_from_files(test_files):
    dict_array=[]
    for test_file in test_files:
        with open("models/"+test_file, "r") as f:
            k = list(zip(*(line.strip().split('\t') for line in f)))
        data = {}
        for column in k:
            data[column[0]]=column[1:]
        dict_array.append(data)
    return dict_array

def make_loss_epoch(test_file, epoch): #"vgg_3/trained_vgg_3_autoencoder_test.txt"
    #Get losses or accuracy from logfile for one epoch, based on the name of the testfile
    #lin-spaced epoch data is added (slightly off)
    loss_epoch_file=test_file[:-8]+"epoch"+str(epoch)+"_log.txt"

    with open("models/"+loss_epoch_file, "r") as f:
        losses=[]
        use_column = 1 #0 is samples, 1 is loss, 2 is accuracy if supervised model
        for line in f:
            if "#" in line:
                if "Accuracy" in line:
                    use_column=2
                continue
            losses.append( float(line.strip().split('\t')[use_column]) )
            
    epoch_points = np.linspace(epoch-1,epoch,len(losses), endpoint=False)
    return [epoch_points, losses]


def make_test_train_plot(epochs, test, train, label, epochs_train, ylabel):
    plt.ylabel(ylabel)
    test_plot = plt.plot(epochs, test, marker="o", label=label)
    plt.plot(epochs_train, train, linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6, label="")
    return test_plot

def generate_data_for_plots(data_dict):
    #array that contains n arrays (n models to plot), each containing [epoch, test value, train value, train epoch, ylabel]
    max_epoch=0
    data_for_all_plots=[]
    for i,data_dict in enumerate(dict_array):
        current_max_epoch = max(tuple(map(int, data_dict["Epoch"])))
        data_array=[]
        if max_epoch<current_max_epoch:
            max_epoch=current_max_epoch 
            
        if "Test acc" in data_dict:
            ylabel="Accuracy (%)"
            train_epoch_data=[]
            train_acc_data=[]
            for epoch in data_dict["Epoch"]:
                e,l = make_loss_epoch(test_files[i], int(epoch))
                train_epoch_data.extend(e)
                train_acc_data.extend(l)
            data_array=[data_dict["Epoch"], data_dict["Test acc"], train_acc_data, modelnames[i], train_epoch_data, ylabel]
        else:
            ylabel="Loss"
            train_epoch_data=[]
            train_loss_data=[]
            for epoch in data_dict["Epoch"]:
                e,l = make_loss_epoch(test_files[i], int(epoch))
                train_epoch_data.extend(e)
                train_loss_data.extend(l)
            data_array=[data_dict["Epoch"], data_dict["Test loss"], train_loss_data, modelnames[i], train_epoch_data, ylabel]
        
        data_for_all_plots.append(data_array)
    return np.array(data_for_all_plots), max_epoch



debug=False
if debug==False:
    #Generate data from files
    dict_array = make_dicts_from_files(test_files)
    data_for_plots, max_epoch=generate_data_for_plots(dict_array)
else:
    # Epoch, test, train, label, trainepoch, ylabel
    data_for_plots, max_epoch=[[np.linspace(0,9,10), np.linspace(0,9,10)*0.1, np.linspace(0,9,10)*0.1 + 1, "Test", np.linspace(0,9,10), "Loss" ], ], 10
    data_for_plots=np.array(data_for_plots*2)
    
fig, ax=plt.subplots()
plt.ylabel(data_for_plots[0][5])

label_array = data_for_plots[:,3]
if len(labels_override) == len(label_array):
    label_array=labels_override
else:
    print("Custom label array does not have the proper length (",len(label_array),"). Using default labels...")

if len(colors) == len(label_array):
    color_override = True
else:
    color_override = False
    print("color array does not have the rights size (", len(label_array), "), using default colors.")

if dump_to_file is not None:
    print("Saving plot data to", dump_to_file)
    np.save(dump_to_file, data_for_plots)

handles1=[]
#plot the data in one plot
for i,data_of_model in enumerate(data_for_plots):
    if color_override==True:
        test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o", color=colors[i])
    else:
        test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o")
    ax.plot(data_of_model[4], data_of_model[2], linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6)
    handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(), lw=3, label=label_array[i])
    handles1.append(handle_for_legend)

#lhandles, llabels = ax.get_legend_handles_labels()
legend1 = plt.legend(handles=handles1, loc=legend_locations[0])

test_line = mlines.Line2D([], [], color='grey', marker="o", label='Test')
train_line = mlines.Line2D([], [], color='grey', linestyle="-", alpha=0.5, lw=2, label='Train')
legend2 = plt.legend(handles=[test_line,train_line], loc=legend_locations[1])

ax.add_artist(legend1)
ax.add_artist(legend2)

plt.xlim((0.2,max_epoch))
#plt.xticks( np.linspace(1,max_epoch,max_epoch) )

if xticks is not None:
    plt.xticks( xticks )
else:
    plt.xticks( np.arange(0,max_epoch+1,10) )

plt.xlabel(xlabel)
plt.title(title)
plt.grid(True)
plt.show()

