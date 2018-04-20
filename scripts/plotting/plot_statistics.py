# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import csv

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

def make_loss_epoch(test_file, epoch): #"vgg_3/trained_vgg_3_autoencoder_test.txt"
    #Get losses or accuracy from logfile for one epoch, based on the name of the testfile
    #lin-spaced epoch data is added (slightly off)
    loss_epoch_file=test_file[:-8]+"epoch"+str(epoch)+"_log.txt"

    data = read_out_file(loss_epoch_file)
    #should have: Batch, Loss (,Accuracy)
    if "Accuracy" in data:
        losses = data["Accuracy"]
    else:
        losses = data["Loss"]
    
    #old version
    """
    with open(loss_epoch_file, "r") as f:
        losses=[]
        use_column = 1 #0 is samples, 1 is loss, 2 is accuracy if supervised model
        for line in f:
            if "#" in line:
                if "Accuracy" in line:
                    use_column=2
                continue
            losses.append( float(line.strip().split('\t')[use_column]) )
    """
    
    epoch_points = np.linspace(epoch-1,epoch,len(losses), endpoint=False)
    return [epoch_points, losses]


def make_test_train_plot(epochs, test, train, label, epochs_train, ylabel):
    #Plot   epochs vs. test   and   epochs_train vs train   , with a unified label
    if ylabel is not None: plt.ylabel(ylabel)
    test_plot = plt.plot(epochs, test, marker="o", label=label)
    plt.plot(epochs_train, train, linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6, label="")
    return test_plot


def make_data_for_plot(dict_array, test_files):
    #Returns ( [[Test_epoch, Test_ydata, Train_epoch, Train_ydata], ...], ylabel_list) for every test file
    return_list_array=[]
    ylabel_list=[]
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
            ylabel = "Accuracy"
            test_acc    = list(map(float, data_dict["Test acc"])) 
            return_list = [test_epochs, test_acc, train_epoch_data, train_y_data]   
        else:
            #This is an autoencoder network
            ylabel = "Loss"
            test_loss    = list(map(float, data_dict["Test loss"])) 
            return_list = [test_epochs, test_loss, train_epoch_data, train_y_data]
            
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
    

def make_data_from_files(test_files, dump_to_file=None):
    #Input: list of strings, each the path to a model in the models folder
    
    # Takes a list of names of test files, and returns:
    # ( [Test_epoch, Test_ydata, Train_epoch, Train_ydata], [...], ... ] , [ylabel1, ylabel2, ...], [default labels] )
    #                 For test file 1,                      File 2 ...   ,   File1 ,  File2,  ...
    #ydata is loss, or instead acc if that is to be found in the log files
    #default labels are the labels for the legend
    dict_array = make_dicts_from_files(test_files) #a list of dicts, one for every test file
    data_from_files, ylabel_list = make_data_for_plot(dict_array, test_files)
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


def make_plot_same_y(data_for_plots, default_label_array, xlabel, ylabel_list, title, legend_locations, labels_override, colors, xticks, figsize): 
    """
    Makes a plot of one or more graphs, each with the same y-axis (e.g. loss, acc)
    """
    
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
        if color_override==True:
            test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o", color=colors[i])
        else:
            test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o")
        #the train plot
        ax.plot(data_of_model[2], data_of_model[3], linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6)
        handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(), lw=3, label=label_array[i])
        handles1.append(handle_for_legend)
        #for proper yrange, look for min/max of ydata, but not for the first epochs train,
        #since loss is often extreme here
        train_epoch_select = np.array(data_of_model[2]).astype(float)
        train_epoch = np.array(data_of_model[3]).astype(float)
        train_epoch_select = train_epoch_select>=3
        y_value_extrema.extend( [max(data_of_model[1]), min(data_of_model[1]),
                                 max(train_epoch[train_epoch_select]), min(train_epoch[train_epoch_select])] )
    
    #lhandles, llabels = ax.get_legend_handles_labels()
    legend1 = plt.legend(handles=handles1, loc=legend_locations[0])
    
    test_line = mlines.Line2D([], [], color='grey', marker="o", label='Test')
    train_line = mlines.Line2D([], [], color='grey', linestyle="-", alpha=0.5, lw=2, label='Train')
    legend2 = plt.legend(handles=[test_line,train_line], loc=legend_locations[1])
    
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    #xrange
    max_epoch = get_max_epoch(data_for_plots)
    plt.xlim((0,max_epoch))

    if xticks is not None:
        plt.xticks( xticks )
    else:
        plt.xticks( np.arange(0, max_epoch+1,5) )

    #yrange
    plt.ylim(get_proper_range(y_value_extrema))

    ax.set_xlabel(xlabel)
    plt.title(title)
    plt.grid(True)
    return(fig)


def make_plot_same_y_parallel(data_autoencoder, data_parallel_train, data_parallel_test, default_label_array, xlabel, ylabel_list, title, legend_locations, labels_override, colors, xticks, figsize, data_parallel_2=None): 
    """
    Makes a plot of autoencoder loss and supervised acc of parallel training.
    data autoencoder :
    [Test Epoch, Test ydata, Epoch train, ydata train]
    and data_parallel:
    [Epoch, ydata]
    """
    fig, ax=plt.subplots(figsize=figsize)
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
    plt.xlim((0,max_epoch))
    
    #y range
    ax.set_ylim(get_proper_range(data_autoencoder[1]))
    ax2.set_ylim(get_proper_range(np.concatenate((data_parallel_test[1],data_parallel_train[1]))))
    
    if xticks is not None:
        plt.xticks( xticks )
    else:
        plt.xticks( np.arange(0, max_epoch+1,5) )
        
    ax.set_ylabel(ylabel_list[0])
    ax2.set_ylabel(ylabel_list[1])
    ax.set_xlabel(xlabel)
    plt.title(title)
    ax.grid(True)
    return(fig)
    



