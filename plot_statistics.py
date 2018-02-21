# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

hd = True
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

    
def make_dicts_from_files(test_files):
    #Read in multiple test files and for each, sort all columns in their own dict entry
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


def make_test_train_plot(epochs, test, train, label, epochs_train=None):
    test_plot = plt.plot(epochs, test, label=label, marker="o")
    if epochs_train==None:
        plt.plot(epochs, train, linestyle="--", color=test_plot[0].get_color(), label="", marker="x")
    else:
        plt.plot(epochs_train, train, linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6, label="")
    
    #x_ticks_major = np.arange(0, 101, 10)
    #plt.xticks(x_ticks_major)
    #plt.minorticks_on()
    #plt.ylim((0, 1))

def make_data_for_plot(dict_array, test_files):
    #Returns [Test_epoch, Test_ydata, Train_epoch, Train_ydata]
    return_list_array=[]
    for i,data_dict in enumerate(dict_array): 
        #Gather training loss from the _log.txt files for every epoch in the test file:
        train_epoch_data=[]
        train_y_data=[]
        for epoch in data_dict["Epoch"]:
            e,l = make_loss_epoch(test_files[i], int(epoch)) #automatically takes acc if available
            train_epoch_data.extend(e)
            train_y_data.extend(l)
                
        test_epochs =      list(map(int,     data_dict["Epoch"])) 
        #train_epoch_data = list(map(float,   train_epoch_data ))
        #train_y_data =     list(map(float,   train_y_data))
        
        if "Test acc" in data_dict:
            #This is an encoder network
            plt.ylabel("Accuracy")
            test_acc    = list(map(float, data_dict["Test acc"])) 
            return_list = [test_epochs, test_acc, train_epoch_data, train_y_data]   
        else:
            #This is an autoencoder network
            plt.ylabel("Loss")
            test_loss    = list(map(float, data_dict["Test loss"])) 
            return_list = [test_epochs, test_loss, train_epoch_data, train_y_data]
            
        return_list_array.append(return_list)
    return return_list_array

def show_the_plots(data_list_array, test_files):
    modelnames=[] #e.g. vgg_3_autoencoder; used for labels in plot
    for modelident in test_files:
        modelnames.append(modelident.split("trained_")[1][:-9])
    
    #one plot for every entry in the array = every test file
    max_epoch_of_all_plots=0
    for i,data_list in enumerate(data_list_array):
        test_epochs = data_list[0]
        max_epoch=max(test_epochs)
        if max_epoch>max_epoch_of_all_plots:
            max_epoch_of_all_plots = max_epoch 
            
        make_test_train_plot(epochs=test_epochs, test=data_list[1], epochs_train=data_list[2], train=data_list[3], label=modelnames[i])
    
    plt.xlim((0.2,max_epoch))
    plt.xticks( np.linspace(1,max_epoch,max_epoch) )
    plt.legend()
    plt.xlabel('Epoch')
    plt.title("Progress")
    plt.grid(True)
    plt.show()

#contains a dict for every test file
dict_array = make_dicts_from_files(test_files)
data_list_array = make_data_for_plot(dict_array, test_files)
show_the_plots(data_list_array, test_files)


"""
max_epoch=0
for i,data_dict in enumerate(dict_array):
    current_max_epoch = max(tuple(map(int, data_dict["Epoch"])))
    if max_epoch<current_max_epoch:
        max_epoch=current_max_epoch 
        
    if "Test acc" in data_dict:
        plt.ylabel("Accuracy")
        if hd == True:
            train_epoch_data=[]
            train_acc_data=[]
            for epoch in data_dict["Epoch"]:
                e,l = make_loss_epoch(test_files[i], int(epoch))
                train_epoch_data.extend(e)
                train_acc_data.extend(l)
            
            make_test_train_plot(data_dict["Epoch"], data_dict["Test acc"], train_acc_data, modelnames[i], epochs_train=train_epoch_data)
       
        else:
            make_test_train_plot(data_dict["Epoch"], data_dict["Test acc"], data_dict["Train acc"], modelnames[i])
        
        
    else:
        plt.ylabel("Loss")
        if hd == True:
            train_epoch_data=[]
            train_loss_data=[]
            for epoch in data_dict["Epoch"]:
                e,l = make_loss_epoch(test_files[i], int(epoch))
                train_epoch_data.extend(e)
                train_loss_data.extend(l)
            
            make_test_train_plot(data_dict["Epoch"], data_dict["Test loss"], train_loss_data, modelnames[i], epochs_train=train_epoch_data)
            
        else:
            make_test_train_plot(data_dict["Epoch"], data_dict["Test loss"], data_dict["Train loss"], modelnames[i])


        
plt.xlim((0.2,max_epoch))
plt.xticks( np.linspace(1,max_epoch,max_epoch) )
plt.legend()
plt.xlabel('Epoch')
plt.title("Progress")
plt.grid(True)
plt.show()
"""