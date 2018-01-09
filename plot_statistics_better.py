# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

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

#vgg4 autoencoders
xtest_files = ["vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
              "vgg_4_ConvAfterPool/trained_vgg_4_ConvAfterPool_autoencoder_test.txt",
              "vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
              "vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
              "vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",]


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
    test_plot = plt.plot(epochs, test, marker="o")
    plt.plot(epochs_train, train, linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6, label="")
    return test_plot[0].get_color()

def generate_data_for_plots(data_dict):
    #array that contains n arrays (n models to plot), each containing [epoch, test value, train value, train epoch, ylabel]
    max_epoch=0
    data_for_all_plots=[]
    for i,data_dict in enumerate(dict_array):
        current_max_epoch = max(tuple(map(int, data_dict["Epoch"])))
        if max_epoch<current_max_epoch:
            max_epoch=current_max_epoch 
            
        if "Test acc" in data_dict:
            ylabel="Accuracy"
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
    return data_for_all_plots, max_epoch

fig, ax=plt.subplots()
#Train and Test legend entries
plt.plot([], [], color="grey", label="Test", marker="o")
plt.plot([], [], linestyle="-", color="grey", alpha=0.5, lw=0.6, label="Train")
handles, labels = ax.get_legend_handles_labels()

debug=False
if debug==False:
    #Generate data from files
    dict_array = make_dicts_from_files(test_files)
    data_for_plots, max_epoch=generate_data_for_plots(dict_array)
else:
    #debug:
    # Epoch, test, train, label, trainepoch, ylabel
    data_for_plots, max_epoch=[[np.linspace(0,9,10), np.linspace(0,9,10)*0.1, np.linspace(0,9,10)*0.1 + 1, "Test", np.linspace(0,9,10), "loss" ], ], 10

#plot the data in one plot
handle_array=[]
for data_of_model in data_for_plots*2:
    color=make_test_train_plot(data_of_model[0], data_of_model[1],data_of_model[2],data_of_model[3],data_of_model[4],data_of_model[5],)
    patch = mpatches.Patch(color=color, label=data_of_model[5])
    handle_array.append(patch)




plt.xlim((0.2,max_epoch))
#plt.xticks( np.linspace(1,max_epoch,max_epoch) )
plt.legend(handles=handle_array+handles)

#legend=plt.legend(handles=handle_array+[handles,], labels+['Epoch of autoencoder', ])

plt.xlabel('Epoch')
plt.title("Progress")
plt.grid(True)
plt.show()

