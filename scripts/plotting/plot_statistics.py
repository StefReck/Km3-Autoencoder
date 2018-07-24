# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import csv

import sys
sys.path.append('/home/woody/capn/mppi013h/Km3-Autoencoder/scripts/')
from util.saved_setups_for_plot_statistics import get_plot_statistics_plot_size

"""
Contains all the utility to read in log files and plot them.
"""    

def read_out_file(logfile_path):
    #Input: Filepath of logfile
    #Output:Contents of this file as a dict, with the first item per column as the keywords
    raw = []
    with open(logfile_path, "r") as f:
        for line in csv.reader(f, delimiter="\t"):
            if len(line) <= 1:
                #Skip notes and empty lines (they dont have tabs)
                continue
            if line[0][0]=="#":
                line[0]=line[0][1:]
            raw.append(line)
    raw=list(zip(*raw))
    data = {}
    for column in raw:   
        data[column[0]]=column[1:]
    return data

def make_dicts_from_files(test_files):
    #Read in multiple test files and for each, sort all columns in their own dict entry
    # returns list of dicts of length len(test_files)
    dict_array=[]
    for test_file in test_files:
        data = read_out_file(test_file)
        dict_array.append(data)
    return dict_array

def make_loss_epoch(test_file, epoch, which_ydata): #"vgg_3/trained_vgg_3_autoencoder_test.txt"
    #Get losses or accuracy from logfile for one epoch, based on the name of the testfile
    #which_ydata is is the label of the column that gets returned
    #usually Accuracy or Loss
    #lin-spaced epoch data is added (slightly off)
    loss_epoch_file=test_file[:-8]+"epoch"+str(epoch)+"_log.txt"

    data = read_out_file(loss_epoch_file)
    #should have: Batch, Loss (,Accuracy)
    losses = data[which_ydata]
    """
    if "Accuracy" in data:
        losses = data["Accuracy"]
    else:
        losses = data["Loss"]
    """
    epoch_points = np.linspace(epoch-1,epoch,len(losses), endpoint=False)
    return [epoch_points, losses]


def make_test_train_plot(epochs, test, train, label, epochs_train, ylabel):
    #Plot   epochs vs. test   and   epochs_train vs train   , with a unified label
    if ylabel is not None: plt.ylabel(ylabel)
    test_plot = plt.plot(epochs, test, marker="o", label=label)
    plt.plot(epochs_train, train, linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6, label="")
    return test_plot


def make_data_for_plot(dict_array, test_files, which_ydata):
    #Returns ( [[Test_epoch, Test_ydata, Train_epoch, Train_ydata], ...], ylabel_list) for every test file
    return_list_array=[]
    ylabel_list=[]
    for i,data_dict in enumerate(dict_array): 
        number_of_test_epochs = len(data_dict["Epoch"])
        which_y=which_ydata[i]
        
        # "which_y" is the column label in the train files
        #default: same ydata for all epochs is taken
        which_ydatas = [which_y,]*number_of_test_epochs
        if which_y=="Accuracy":
            test_file_column_labels = ["Test acc",]*number_of_test_epochs
            ylabel = "Accuracy"
            
        elif which_y=="cat_cross_inv":
            #For adversarial AE, critic and generator training is alternated evey epoch,
            #so a different label has to be taken every 2nd epoch
            test_file_column_labels = ["Test cat_cross_inv",]*number_of_test_epochs
            for j in range(1,number_of_test_epochs,2):
                which_ydatas[j]="Loss"
                test_file_column_labels[j]="Test loss"
            ylabel = "loss inv"
            
        else:
            #just take the loss instead
            test_file_column_labels = ["Test loss",]*number_of_test_epochs
            ylabel = "Loss"
            
        test_epochs, test_ydata = [], []
        train_epoch_data, train_y_data = [], []
        
        for test_file_line, test_file_epoch in enumerate(data_dict["Epoch"]):
            which_y = which_ydatas[test_file_line]
            test_file_column_label = test_file_column_labels[test_file_line]
            
            test_epochs.append(int(test_file_epoch))
            test_ydata.append(float(data_dict[test_file_column_label][test_file_line]))
            
            e,l = make_loss_epoch(test_files[i], int(test_file_epoch), which_y)
            train_epoch_data.extend(e)
            train_y_data.extend(l)
        
        #Append to list for every test file given
        return_list = [test_epochs, test_ydata, train_epoch_data, train_y_data]
        return_list_array.append(return_list)
        ylabel_list.append(ylabel)
        
    return return_list_array, ylabel_list

def get_max_epoch(data_from_files):
    # Goes through the data list from multiple test files, and looks up the highest
    # epoch from all, so this is the highest epoch that will be visible in the plot
    max_epoch_of_all_curves=0
    for i,data_list in enumerate(data_from_files):
        test_epochs = data_list[0]
        max_epoch=max(test_epochs)
        if max_epoch>max_epoch_of_all_curves:
            max_epoch_of_all_curves = max_epoch 
    return max_epoch_of_all_curves

def get_default_labels(test_files):
    # Generate default labels for every test file for plotting, e.g.:
    # INPUT   vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt
    # OUTPUT                    vgg_3_eps_autoencoder
    modelnames=[] 
    for modelident in test_files:
        modelnames.append(modelident.split("trained_")[1][:-9])
    return modelnames
    

def make_data_from_files(test_files, which_ydata="auto", dump_to_file=None):
    #Input: list of strings, each the path to a model in the models folder
    
    # Takes a list of names of test files, and returns:
    # ( [Test_epoch, Test_ydata, Train_epoch, Train_ydata], [...], ... ] , [ylabel1, ylabel2, ...], [default labels] )
    #                 For test file 1,                      File 2 ...   ,   File1 ,  File2,  ...
    #which ydata is returned is defined by the name of which_ydata, the name of the loss
    #which is also the top line of the train log files, e.g. Accuracy, Loss, cat_cross_inv
    #default labels are the labels for the legend
    
    #a list of dicts, one for every test file, conatining all columns of the testfile
    dict_array = make_dicts_from_files(test_files) 
    
    if which_ydata=="auto":
        #take acc if available, loss otherwise
        which_ydata=[]
        for dictx in dict_array:
            if "Test acc" in dictx:
                which_ydata.append("Accuracy")
            elif "Test cat_cross_inv" in dictx:
                which_ydata.append("cat_cross_inv")
            else:
                which_ydata.append("Loss")
    
    #data from the single epoch train files, together with the test file data
    data_from_files, ylabel_list = make_data_for_plot(dict_array, test_files, which_ydata)
    #labels for plot
    default_label_array = get_default_labels(test_files)
    
    if dump_to_file is not None:
        save_path = "results/dumped_statistics/"+dump_to_file
        print("Saving plot data via np.save to", save_path)
        print("This file contains (data_from_files, ylabel_list, default_label_array)")
        data_to_be_saved = (data_from_files, ylabel_list, default_label_array)
        np.save(save_path, data_to_be_saved)
    
    return data_from_files, ylabel_list, default_label_array

    
def get_proper_range(ydata, relative_spacing=(0.05, 0.2)):
    mini = min(np.ravel(ydata).astype(float))
    maxi = max(np.ravel(ydata).astype(float))
    span = maxi - mini
    ranges = (mini-span*relative_spacing[0], maxi+span*relative_spacing[1])
    return ranges



def get_last_prl_epochs(data_autoencoder, data_parallel, how_many_epochs_each_to_train):
    #Input: data from an autoenocoder and the parallel training that was made with it
    #       how many epochs of parallel training were done on each AE epoch
    #       data_parallel = [test_epochs, test_yacc, train_epochs, train_acc]
    
    #Output:train and test data of the last prl epoch that was trained on each AE epoch
    #           each containing [epoch, ydata]

    
    #The test data:
    #how_many_epochs_each_to_train = e.g. [10,2,2,2,2,2,1,1,...]
    take_these_prl_epochs=np.cumsum(how_many_epochs_each_to_train)
    #only take epochs that have actually been made:
    highest_prl_epoch = max(data_parallel[0])
    #if the highest prl epoch was e.g. 21, take_these_p will now be [10,12,14,16,18,20,21]
    take_these_prl_epochs=np.array(take_these_prl_epochs[take_these_prl_epochs<=highest_prl_epoch])
    #contains [epoch, ydata]
    data_parallel_test_ydata = np.array(data_parallel[1])[take_these_prl_epochs-1]
    data_parallel_test_epoch = np.arange(1, len(data_parallel_test_ydata)+1)
    data_parallel_test = [data_parallel_test_epoch, data_parallel_test_ydata]
    #formerly, it contained the epoch of the supervised training, and not the one from the autoencoder,
    #which it will be plotted against later:
    #data_parallel_test = np.array(data_parallel[0:2])[:,take_these_prl_epochs-1]


    #train: Only take epochs that were trained for one Epoch on an AE Epoch
    #how_many_epochs e.g. [10,2,2,1,1]
    #take_these_prl e.g.  [10,12,14,15,16]
    for index,(i, j) in enumerate(zip(take_these_prl_epochs[:-1], take_these_prl_epochs[1:])):
        if (j-i)==1:
            is_1=take_these_prl_epochs[index:] #e.g. [15,16]
            #shift epochs, so that it will be plotted over the AE epoch and not the spvsd epoch
            shift_epochs_by = is_1[0]-(index+1) #e.g. 15-(3+1)=11
            break
    """
    print("take_these_prl_epochs", take_these_prl_epochs)
    print("is_1",is_1)
    print("shift_epochs_by",shift_epochs_by)
    """
    #data_parallel_train=[train_epoch, train_ydata]
    data_parallel_train=[[],[]]
    for epoch in is_1:
        take_these = np.logical_and(data_parallel[2]>=epoch, data_parallel[2]<epoch+1)
        data_parallel_train[0].extend( (np.array(data_parallel[2])-shift_epochs_by)[take_these])
        data_parallel_train[1].extend( np.array(data_parallel[3])[take_these])
    
    return np.array(data_parallel_test), np.array(data_parallel_train)

def print_extrema(epochs, ydata):
    #Print the maximum and the minimum of the ydata, together with their epoch
    maximum_epoch = np.argmax(ydata)
    minimum_epoch = np.argmin(ydata)
    print("Test data extrema:")
    print("Maximum:\t",epochs[maximum_epoch],"\t",ydata[maximum_epoch])
    print("Minimum:\t",epochs[minimum_epoch],"\t",ydata[minimum_epoch])
    return

def make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, legend_locations, labels_override, colors, xticks, style, xrange="auto", average_train_data_bins=1): 
    """
    Makes a plot of one or more graphs, each with the same y-axis (e.g. loss, acc)
    average_train_data_bins: take the average of every two entries in the
                            train data (less jitter)
    """
    figsize, font_size = get_plot_statistics_plot_size(style)
    plt.rcParams.update({'font.size': font_size})
    fig, ax=plt.subplots(figsize=figsize)
    
    all_ylabels_equal = all(x == ylabel_list[0] for x in ylabel_list)
    if all_ylabels_equal == False:
        print("Warning: Not all ylabels are equal:", ylabel_list, ",\nchoosing ", ylabel_list[0] )
    
    ylabel = ylabel_list[0]
    plt.ylabel(ylabel)
    
    if len(labels_override) == len(default_label_array):
        label_array=labels_override
    else:
        label_array = default_label_array
        print("Custom label array does not have the proper length (",len(label_array),"). Using default labels...")
        
    if len(colors) == len(label_array):
        color_override = True
    else:
        color_override = False
        print("color array does not have the rights size (", len(label_array), "), using default colors.")
    
    handles1=[]
    y_value_extrema=[]
    #plot the data in one plot
    for i,data_of_model in enumerate(data_for_plots):
        # data_of_model: [Test_epoch, Test_ydata, Train_epoch, Train_ydata]
        train_epoch = np.array(data_of_model[2]).astype(float)
        train_ydata = np.array(data_of_model[3]).astype(float)
        
        print(label_array[i])
        print_extrema(data_of_model[0],data_of_model[1])
        
        if average_train_data_bins != 1:
            #assure len(train data) is devisable by average_train_bins
            rest = len(train_ydata)%average_train_data_bins
            if rest != 0:
                train_ydata=train_ydata[:-rest]
                train_epoch=train_epoch[:-rest]
            #average over every average_train_bins bins
            train_ydata = np.mean(train_ydata.reshape(-1, average_train_data_bins), axis=1)
            train_epoch = np.mean(train_epoch.reshape(-1, average_train_data_bins), axis=1)
        
        if color_override==True:
            test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o", color=colors[i])
        else:
            test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o")
        #the train plot
        ax.plot(train_epoch, train_ydata, linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6)
        handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(), lw=3, label=label_array[i])
        handles1.append(handle_for_legend)
        #for proper yrange, look for min/max of ydata, but not for the first epochs train,
        #since loss is often extreme here
        take_range_after_epoch = 3
        if len(train_ydata[train_epoch>=take_range_after_epoch])==0:
            #epoch 3 doesnt exist
            take_range_after_epoch=0
            
        y_value_extrema.extend([max(data_of_model[1]), 
                                min(data_of_model[1]),
                                max(train_ydata[train_epoch>=take_range_after_epoch]), 
                                min(train_ydata[train_epoch>=take_range_after_epoch]) ])
    
    #lhandles, llabels = ax.get_legend_handles_labels()
    legend1 = plt.legend(handles=handles1, loc=legend_locations[0])
    
    test_line = mlines.Line2D([], [], color='grey', marker="o", label='Test')
    train_line = mlines.Line2D([], [], color='grey', linestyle="-", alpha=0.5, lw=2, label='Train')
    legend2 = plt.legend(handles=[test_line,train_line], loc=legend_locations[1])
    
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    #xrange
    if xrange=="auto":
        max_epoch = get_max_epoch(data_for_plots)
        ax.set_xlim((0,max_epoch))
    else:
        max_epoch = max(xrange)
        ax.set_xlim(xrange)

    if xticks is not None:
        plt.xticks( xticks )
    else:
        if max_epoch>=100:
            increment = 10
        else:
            increment = 5
        plt.xticks( np.arange(0, max_epoch+1,increment) )

    #yrange
    plt.ylim(get_proper_range(y_value_extrema))

    ax.set_xlabel(xlabel)
    plt.title(title)
    plt.grid(True)
    return(fig)


def make_plot_same_y_parallel(data_autoencoder, data_parallel_train, data_parallel_test, default_label_array, xlabel, ylabel_list, title, legend_locations, labels_override, colors, xticks, style, data_parallel_2=None, ylims=None, AE_yticks=None): 
    """
    Makes a plot of autoencoder loss and supervised acc of parallel training.
    data autoencoder :
    [Test Epoch, Test ydata, Epoch train, ydata train]
    and data_parallel_test/_train:
    [Epoch, ydata]
    """
    figsize, font_size = get_plot_statistics_plot_size(style)
    plt.rcParams.update({'font.size': font_size})
    fig, ax=plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.15)
    
    ax2 = ax.twinx()
    
    if len(labels_override) == len(default_label_array):
        label_array=labels_override
    else:
        label_array = default_label_array
        print("Custom label array does not have the proper length (",len(label_array),"). Using default labels...")
        
    
    if len(colors) == len(label_array):
        color_override = True
    else:
        color_override = False
        print("color array does not have the rights size (", len(label_array), "), using default colors.")
    
    
    #plot the data in one plot
    #autoencoder
    if color_override==True:
        test_plot = ax.plot(data_autoencoder[0], data_autoencoder[1], marker="o", color=colors[0], zorder=10)
    else:
        test_plot = ax.plot(data_autoencoder[0], data_autoencoder[1], marker="o", zorder=10)
    
    #the train plot
    ax.plot(data_autoencoder[2], data_autoencoder[3], linestyle="-", 
            color=test_plot[0].get_color(), alpha=0.5, lw=0.6, zorder=1)
    
    
    #parallel enc
    #parallel training might not have been done for all AE epochs:
    #data_parallel_epochs = data_autoencoder[0][:len(data_parallel_test[0])]
    print_extrema(data_parallel_test[0], data_parallel_test[1])
    if color_override==True:#        21                      22
        test_plot_prl = ax2.plot(data_parallel_test[0], data_parallel_test[1], marker="o", color=colors[1], zorder=11)
    else:
        test_plot_prl = ax2.plot(data_parallel_test[0], data_parallel_test[1], marker="o", zorder=11)
    
    #train plot
    ax2.plot(data_parallel_train[0], data_parallel_train[1], linestyle="-", 
            color=test_plot_prl[0].get_color(), alpha=0.5, lw=0.6, zorder=2)
    
    #custom handles for all the lines of both axes
    handle_for_legend_array=[]
    
    if data_parallel_2 != None:
        #Optional: A second parallel plot is added
        if color_override==True:
            test_plot_prl_2 = ax2.plot(data_parallel_2[0], data_parallel_2[1], marker="o", color=colors[2])
        else:
            test_plot_prl_2 = ax2.plot(data_parallel_2[0], data_parallel_2[1], marker="o")
        ax2.plot(data_parallel_2[2], data_parallel_2[3], linestyle="-", 
            color=test_plot_prl_2[0].get_color(), alpha=0.5, lw=0.6)
        handle_for_legend_prl_2 = mlines.Line2D([], [], color=test_plot_prl_2[0].get_color(), 
                                      lw=3, label=label_array[2])
        handle_for_legend_array.append(handle_for_legend_prl_2)
        
    
    handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(),
                                      lw=3, label=label_array[0])
    handle_for_legend_prl = mlines.Line2D([], [], color=test_plot_prl[0].get_color(), 
                                      lw=3, label=label_array[1])
    handle_for_legend_array.extend([handle_for_legend_prl,handle_for_legend])
    legend1 = ax.legend(handles=handle_for_legend_array, loc=legend_locations[0])
    ax.add_artist(legend1)
    
    
    #the test/train box
    test_line = mlines.Line2D([], [], color='grey', marker="o", label='Test')
    train_line = mlines.Line2D([], [], color='grey', linestyle="-", alpha=0.5, 
                               lw=2, label='Train')
    legend2 = ax.legend(handles=[test_line,train_line], loc=legend_locations[1])
    ax.add_artist(legend2)
    
    #x range
    max_epoch = get_max_epoch( [data_autoencoder, data_parallel_test] )
    ax.set_xlim((0,max_epoch))
    #y range
    if ylims==None:
        ylims_AE = get_proper_range(data_autoencoder[1])
        ylims_prl = get_proper_range(get_proper_range(np.concatenate((data_parallel_test[1],data_parallel_train[1]))))
    else:
        ylims_AE, ylims_prl = ylims
    ax.set_ylim(ylims_AE)
    ax2.set_ylim(ylims_prl)
    
    if xticks is not None:
        ax.set_xticks( xticks )
    else:
        if max_epoch<100:
            ax.set_xticks( np.arange(0, max_epoch+1,10) )
        else:
            ax.set_xticks( np.arange(0, max_epoch+1,20) )
    if AE_yticks != None:
        ax.set_yticks(AE_yticks)
        
    ax.set_ylabel(ylabel_list[0]+" (Autoencoder)")
    ax2.set_ylabel(ylabel_list[1]+" (Encoder)")
    ax.set_xlabel(xlabel)
    
    
    plt.title(title)
    ax.grid(True)
    plt.gcf().subplots_adjust(right=0.86)
    return(fig)

