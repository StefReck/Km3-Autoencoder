# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

hd = True
#Test Files to make plots from; Format: vgg_3/trained_vgg_3_autoencoder_test.txt
test_files = ["vgg_3/trained_vgg_3_autoencoder_test.txt", 
              "vgg_3_dropout/trained_vgg_3_dropout_autoencoder_test.txt",
              "vgg_3_max/trained_vgg_3_max_autoencoder_test.txt",
              "vgg_3_stride/trained_vgg_3_stride_autoencoder_test.txt",
              "vgg_3_stride_noRelu/trained_vgg_3_stride_noRelu_autoencoder_test.txt"]

test_files = ["vgg_3/trained_vgg_3_supervised_up_down_test.txt",
              "vgg_3_dropout/trained_vgg_3_dropout_supervised_up_down_test.txt",
              "vgg_3_max/trained_vgg_3_max_supervised_up_down_test.txt",
              "vgg_3_stride/trained_vgg_3_stride_supervised_up_down_test.txt",]

#For debugging
#test_files = ["trained_vgg_3_autoencoder_test.txt", "trained_vgg_3_dropout_autoencoder_test.txt"]

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


dict_array = make_dicts_from_files(test_files)

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