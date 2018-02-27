# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

"""
Contains all the utility to read in log files and plot them.
"""    


def make_dicts_from_files(test_files):
    #Read in multiple test files and for each, sort all columns in their own dict entry
    # returns list of dicts of length len(test_files)
    dict_array=[]
    for test_file in test_files:
        with open(test_file, "r") as f:
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

    with open(loss_epoch_file, "r") as f:
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
    
"""
def show_the_plots(data_list_array, test_files):
    modelnames = get_default_labels(test_files)
    #one plot for every entry in the array = every test file
    max_epoch = get_max_epoch(data_list_array)
    for i,data_list in enumerate(data_list_array):
        make_test_train_plot(epochs=data_list[0], test=data_list[1], epochs_train=data_list[2], train=data_list[3], label=modelnames[i])
    
    plt.xlim((0.2,max_epoch))
    plt.xticks( np.linspace(1,max_epoch,max_epoch) )
    plt.legend()
    plt.xlabel('Epoch')
    plt.title("Progress")
    plt.grid(True)
    plt.show()
"""

def make_data_from_files(test_files, dump_to_file=None):
    # Takes a list of names from the test files, and returns:
    # ( [Test_epoch, Test_ydata, Train_epoch, Train_ydata], [...], ... ] , [ylabel1, ylabel2, ...] )
    #                 For test file 1,                      File 2 ...   ,   File1 ,  File2,  ...
    dict_array = make_dicts_from_files(test_files) #a list of dicts, one for every test file
    data_from_files, ylabel_list = make_data_for_plot(dict_array, test_files)
    
    if dump_to_file is not None:
        print("Saving plot data to", dump_to_file, "via np.save")
        print("This file contains (data_from_files, ylabel_list)")
        data_to_be_saved = (data_from_files, ylabel_list)
        np.save(dump_to_file, data_to_be_saved)
    
    return data_from_files, ylabel_list


def make_plot_same_y(test_files, data_for_plots, xlabel, ylabel_list, title, legend_locations, labels_override, colors, xticks, figsize): 
    fig, ax=plt.subplots(figsize=figsize)
    
    all_ylabels_equal = all(x == ylabel_list[0] for x in ylabel_list)
    if all_ylabels_equal == False:
        print("Warning: Not all ylabels are equal:", ylabel_list, ",\nchoosing ", ylabel_list[0] )
    
    ylabel = ylabel_list[0]
    plt.ylabel(ylabel)
    
    label_array = get_default_labels(test_files)
    if len(labels_override) == len(label_array):
        label_array=labels_override
    else:
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
        # [Test_epoch, Test_ydata, Train_epoch, Train_ydata]
        if color_override==True:
            test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o", color=colors[i])
        else:
            test_plot = ax.plot(data_of_model[0], data_of_model[1], marker="o")
        #the train plot
        ax.plot(data_of_model[2], data_of_model[3], linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6)
        handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(), lw=3, label=label_array[i])
        handles1.append(handle_for_legend)
        y_value_extrema.extend( [max(data_of_model[1]), min(data_of_model[1])] )
    
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
        plt.xticks( np.arange(0, max_epoch+1,10) )
    
    #yrange
    plt.ylim(get_proper_range(y_value_extrema))
    
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True)
    return(fig)
    
def get_proper_range(ydata, relative_spacing=(0.05, 0.2)):
    mini = min(ydata)
    maxi = max(ydata)
    span = maxi - mini
    ranges = (mini-span*relative_spacing[0], maxi+span*relative_spacing[1])
    return ranges

def make_plot_same_y_parallel(test_files, data_autoencoder, data_parallel, xlabel, ylabel_list, title, legend_locations, labels_override, colors, xticks, figsize): 
    fig, ax=plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    
    
    label_array = get_default_labels(test_files)
    if len(labels_override) == len(label_array):
        label_array=labels_override
    else:
        print("Custom label array does not have the proper length (",len(label_array),"). Using default labels...")
    
    if len(colors) == len(label_array):
        color_override = True
    else:
        color_override = False
        print("color array does not have the rights size (", len(label_array), "), using default colors.")
    
    
    #plot the data in one plot
    #autoencoder
    if color_override==True:
        test_plot = ax.plot(data_autoencoder[0], data_autoencoder[1], marker="o", color=colors[0])
    else:
        test_plot = ax.plot(data_autoencoder[0], data_autoencoder[1], marker="o")
    
    #the train plot
    ax.plot(data_autoencoder[2], data_autoencoder[3], linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6)
    
    
    #parallel, no train plot
    #parallel training might not have been done for all AE epochs:
    data_parallel_epochs = data_autoencoder[0][:len(data_parallel[0])]
    
    if color_override==True:
        test_plot_prl = ax2.plot(data_parallel_epochs, data_parallel[1], marker="o", color=colors[1])
    else:
        test_plot_prl = ax2.plot(data_parallel_epochs, data_parallel[1], marker="o")
    
    handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(),
                                      lw=3, label=label_array[0])
    handle_for_legend_prl = mlines.Line2D([], [], color=test_plot_prl[0].get_color(), 
                                      lw=3, label=label_array[1])
    legend1 = ax.legend(handles=[handle_for_legend, handle_for_legend_prl], 
                         loc=legend_locations[0])
    ax.add_artist(legend1)
    
    
    #the test/train box
    test_line = mlines.Line2D([], [], color='grey', marker="o", label='Test')
    train_line = mlines.Line2D([], [], color='grey', linestyle="-", alpha=0.5, 
                               lw=2, label='Train')
    legend2 = ax.legend(handles=[test_line,train_line], loc=legend_locations[1])
    ax.add_artist(legend2)
    
    #x range
    max_epoch = get_max_epoch( [data_autoencoder, data_parallel] )
    plt.xlim((0,max_epoch))
    
    #y range
    ax.set_ylim(get_proper_range(data_autoencoder[1]))
    ax2.set_ylim(get_proper_range(data_parallel[1]))
    
    if xticks is not None:
        plt.xticks( xticks )
    else:
        plt.xticks( np.arange(0, max_epoch+1,10) )
        
    ax.set_ylabel(ylabel_list[0])
    ax2.set_ylabel(ylabel_list[1])
    plt.xlabel(xlabel)
    plt.title(title)
    ax.grid(True)
    return(fig)
    


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