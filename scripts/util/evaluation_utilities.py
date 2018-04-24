#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility code for the evaluation of a network's performance after training.
Copied, added is_autoencoder
xrange -> range (python 3)
steps -> int(steps)
"""

import h5py
import numpy as np
#import keras as ks
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from keras.models import load_model
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import sys
sys.path.append('../')

from util.run_cnn import generate_batches_from_hdf5_file, load_zero_center_data, h5_get_number_of_rows
from get_dataset_info import get_dataset_info


#------------- Functions used in evaluating the performance of model -------------#

def make_performance_array_energy_correct(model, f, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, dataset_info_dict, broken_simulations_mode=0, samples=None):
    """
    Creates an energy_correct array based on test_data that specifies for every event, if the model's prediction is True/False.
    :param ks.model.Model/Sequential model: Fully trained Keras model of a neural network.
    :param str f: Filepath of the file that is used for making predctions.
    :param tuple n_bins: The number of bins for each dimension (x,y,z,t) in the testfile.
    :param (int, str) class_type: The number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int batchsize: Batchsize that should be used for predicting.
    :param ndarray xs_mean: mean_image of the x dataset if zero-centering is enabled.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped in the generator.
    :param None/int samples: Number of events that should be predicted. If samples=None, the whole file will be used.
    :return: ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc and y_pred info for each event.
    """
    # TODO only works for a single test_file till now
    generator = generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean, f_size=None , is_autoencoder=False, yield_mc_info=True, swap_col=swap_4d_channels, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict) # f_size=samples prob not necessary
    
    if samples is None: samples = len(h5py.File(f, 'r')['y'])
    steps = samples/batchsize

    arr_energy_correct = None
    for s in range(int(steps)):
        if s % 300 == 0:
            print ('Predicting in step ' + str(s) + "/" + str(int(steps)))
        xs, y_true, mc_info = next(generator)
        y_pred = model.predict_on_batch(xs)

        # check if the predictions were correct
        correct = check_if_prediction_is_correct(y_pred, y_true)
        energy = mc_info[:, 2]
        particle_type = mc_info[:, 1]
        is_cc = mc_info[:, 3]

        ax = np.newaxis

        # make a temporary energy_correct array for this batch
        arr_energy_correct_temp = np.concatenate([energy[:, ax], correct[:, ax], particle_type[:, ax], is_cc[:, ax], y_pred], axis=1)

        if arr_energy_correct is None:
            arr_energy_correct = np.zeros((int(steps) * batchsize, arr_energy_correct_temp.shape[1:2][0]), dtype=np.float32)
        arr_energy_correct[s*batchsize : (s+1) * batchsize] = arr_energy_correct_temp


    total_accuracy=np.sum(arr_energy_correct[:,1])/len(arr_energy_correct[:,1])
    print("Total accuracy:", total_accuracy, "(", np.sum(arr_energy_correct[:,1]), "of",len(arr_energy_correct[:,1]) , "events)")
    return arr_energy_correct

def make_loss_array_energy_correct(model, f, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, dataset_info_dict, broken_simulations_mode=0, samples=None):
    """
    Creates an energy_correct array based on test_data that specifies for every event the loss of the autoencoder.
    :param ks.model.Model/Sequential model: Fully trained Keras model of a neural network.
    :param str f: Filepath of the file that is used for making predctions.
    :param tuple n_bins: The number of bins for each dimension (x,y,z,t) in the testfile.
    :param (int, str) class_type: The number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int batchsize: Batchsize that should be used for predicting.
    :param ndarray xs_mean: mean_image of the x dataset if zero-centering is enabled.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped in the generator.
    :param None/int samples: Number of events that should be predicted. If samples=None, the whole file will be used.
    :return: ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc and y_pred info for each event.
    """
    # TODO only works for a single test_file till now
    generator = generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean, f_size=None , is_autoencoder=True, yield_mc_info=True, swap_col=swap_4d_channels, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict)
    
    if samples is None: samples = len(h5py.File(f, 'r')['y'])
    steps = samples/batchsize
    
    arr_energy_correct = None
    for s in range(int(steps)):
        if s % 300 == 0:
            print ('Predicting in step ' + str(s) + "/" + str(int(steps)))
        xs, xs_2, mc_info  = next(generator)
        y_pred = model.predict_on_batch(xs) #(32, 11, 18, 50, 1) 

        # calculate mean squared error between original image and autoencoded one
        mse = ((xs - y_pred) ** 2).mean(axis=(1,2,3,4)) #(32,)
        energy = mc_info[:, 2] # (32,)

        ax = np.newaxis

        # make a temporary energy_correct array for this batch, (32, 4)
        arr_energy_correct_temp = np.concatenate([energy[:, ax], mse[:,ax]], axis=1)

        if arr_energy_correct is None:
            arr_energy_correct = np.zeros((int(steps) * batchsize, arr_energy_correct_temp.shape[1:2][0]), dtype=np.float32)
        arr_energy_correct[s*batchsize : (s+1) * batchsize] = arr_energy_correct_temp

    return arr_energy_correct


def check_if_prediction_is_correct(y_pred, y_true):
    """
    Checks if the predictions in y_pred are true.
    E.g. y_pred = [0.1, 0.1, 0.8] ; y_true = [0,0,1] -> Correct.
    Warning: There's a loophole if the prediction is not definite, e.g. y_pred = [0.4, 0.4, 0.2].
    :param ndarray(ndim=2) y_pred: 2D array that contains the predictions of a network on a number of events.
                                   Shape=(#events, n_classes).
    :param ndarray(ndim=2) y_true: 2D array that contains the true classes for the events. Shape=(#events, n_classes).
    :return: ndarray(ndim=1) correct: 1D array that specifies if the prediction for the single events is correct (True) or False.
    """
    # TODO loophole if pred has two or more max values per row
    class_pred = np.argmax(y_pred, axis=1)
    class_true = np.argmax(y_true, axis=1)

    correct = np.equal(class_pred, class_true)
    return correct

def make_autoencoder_energy_data(model, f, n_bins, class_type, batchsize, xs_mean, swap_4d_channels, samples=None):
    """
    Creates an energy_correct array based on test_data that specifies for every event the loss of the autoencoder.
    :param ks.model.Model/Sequential model: Fully trained Keras model of a neural network.
    :param str f: Filepath of the file that is used for making predctions.
    :param tuple n_bins: The number of bins for each dimension (x,y,z,t) in the testfile.
    :param (int, str) class_type: The number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int batchsize: Batchsize that should be used for predicting.
    :param ndarray xs_mean: mean_image of the x dataset if zero-centering is enabled.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped in the generator.
    :param None/int samples: Number of events that should be predicted. If samples=None, the whole file will be used.
    :return: ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc and y_pred info for each event.
    """
    # TODO only works for a single test_file till now
    generator = generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean, f_size=None , is_autoencoder=True, yield_mc_info=True, swap_col=swap_4d_channels) # f_size=samples prob not necessary
    
    if samples is None: samples = len(h5py.File(f, 'r')['y'])
    steps = samples/batchsize
    
    arr_energy_correct = None
    for s in range(int(steps)):
        if s % 300 == 0:
            print ('Predicting in step ' + str(s) + "/" + str(int(steps)))
        xs, xs_2, mc_info  = next(generator)
        y_pred = model.predict_on_batch(xs) #(32, 11, 18, 50, 1) 

        # calculate mean squared error between original image and autoencoded one
        mse = ((xs - y_pred) ** 2).mean(axis=(1,2,3,4)) #(32,)
        energy = mc_info[:, 2] # (32,)

        ax = np.newaxis

        # make a temporary energy_correct array for this batch, (32, 4)
        arr_energy_correct_temp = np.concatenate([energy[:, ax], mse[:,ax]], axis=1)

        if arr_energy_correct is None:
            arr_energy_correct = np.zeros((int(steps) * batchsize, arr_energy_correct_temp.shape[1:2][0]), dtype=np.float32)
        arr_energy_correct[s*batchsize : (s+1) * batchsize] = arr_energy_correct_temp


    #sort array by  ascending energy
    #arr_energy_correct = arr_energy_correct[arr_energy_correct[:,0].argsort()]
    # Calculate loss in energy range
    energy = arr_energy_correct[:, 0]
    losses = arr_energy_correct[:, 1]
    
    bins=np.linspace(3,100,98)
    hist_energy_losses=np.zeros((len(bins)-1))
    bin_indices = np.digitize(energy,bins) #in which bin the lines belong, e.g. [1,1,2,2,2,...], 1-->bin 3-4
    for bin_no in range(min(bin_indices), max(bin_indices)+1):
        hist_energy_losses[bin_no-1] = np.mean(losses[bin_indices==bin_no])
    hist_1d_energy_loss_bins_centered = np.linspace(3.5,99.5,97)
    
    #How many events are in each bin:
    #events_in_bin=np.bincount(bin_indices)
    #plt.step(bins[1:],events_in_bin[1:], where="mid")

    return [hist_1d_energy_loss_bins_centered, hist_energy_losses]


#------------- Functions used in evaluating the performance of model -------------#


#------------- Functions used in making Matplotlib plots -------------#

def make_energy_to_accuracy_data(arr_energy_correct, plot_range=(3, 100), bins=97):
    """
    Makes data for a mpl step plot with Energy vs. Accuracy based on a [Energy, correct] array.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, ptype, is_cc, y_pred].
    :param str title: Title of the mpl step plot.
    :param str filepath: Filepath of the resulting plot.
    :param (int, int) plot_range: Plot range that should be used in the step plot. E.g. (3, 100) for 3-100GeV Data.
    """
    # Calculate accuracy in energy range
    energy = arr_energy_correct[:, 0]
    correct = arr_energy_correct[:, 1]

    hist_1d_energy = np.histogram(energy, bins=bins, range=plot_range) #häufigkeit von energien
    hist_1d_energy_correct = np.histogram(arr_energy_correct[correct == 1, 0], bins=bins, range=plot_range) #häufigkeit von richtigen energien

    bin_edges = hist_1d_energy[1]
    hist_1d_energy_accuracy_bins = np.divide(hist_1d_energy_correct[0], hist_1d_energy[0], dtype=np.float32) #rel häufigkeit von richtigen energien
    
    #For proper plotting with plt.step where="post"
    hist_1d_energy_accuracy_bins=np.append(hist_1d_energy_accuracy_bins, hist_1d_energy_accuracy_bins[-1])
    
    return [bin_edges, hist_1d_energy_accuracy_bins]


def make_energy_to_loss_data(arr_energy_correct, plot_range=(3, 100), bins=97):
    # Calculate loss in energy range
    energy = arr_energy_correct[:, 0]
    losses = arr_energy_correct[:, 1]
    
    bins_list=np.linspace(plot_range[0],plot_range[1],bins)
    hist_energy_losses=np.zeros((len(bins_list)-1))
    bin_indices = np.digitize(energy,bins_list) #in which bin the lines belong, e.g. [1,1,2,2,2,...], 1-->bin 3-4
    for bin_no in range(min(bin_indices), max(bin_indices)+1):
        hist_energy_losses[bin_no-1] = np.mean(losses[bin_indices==bin_no])
        
    #For proper plotting with plt.step where="post"
    hist_energy_losses=np.append(hist_energy_losses, hist_energy_losses[-1])
    

    return [bins_list, hist_energy_losses]


def make_energy_to_accuracy_plot(arr_energy_correct, title, filepath, plot_range=(3, 100)):
    """
    Makes a mpl step plot with Energy vs. Accuracy based on a [Energy, correct] array.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, ptype, is_cc, y_pred].
    :param str title: Title of the mpl step plot.
    :param str filepath: Filepath of the resulting plot.
    :param (int, int) plot_range: Plot range that should be used in the step plot. E.g. (3, 100) for 3-100GeV Data.
    """
    # Calculate accuracy in energy range
    energy = arr_energy_correct[:, 0]
    correct = arr_energy_correct[:, 1]

    hist_1d_energy = np.histogram(energy, bins=98, range=plot_range) #häufigkeit von energien
    hist_1d_energy_correct = np.histogram(arr_energy_correct[correct == 1, 0], bins=98, range=plot_range) #häufigkeit von richtigen energien

    bin_edges = hist_1d_energy[1]
    hist_1d_energy_accuracy_bins = np.divide(hist_1d_energy_correct[0], hist_1d_energy[0], dtype=np.float32) #rel häufigkeit von richtigen energien
    # For making it work with matplotlib step plot
    #hist_1d_energy_accuracy_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_bins))
    bin_edges_centered = bin_edges[:-1] + 0.5

    plt_bar_1d_energy_accuracy = plt.step(bin_edges_centered, hist_1d_energy_accuracy_bins, where='mid')

    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel('Energy [GeV]')
    plt.ylabel('Accuracy')
    plt.ylim((0.5, 1))
    plt.title(title)
    plt.grid(True)

    plt.savefig(filepath+".pdf")

def make_energy_to_accuracy_plot_comp(arr_energy_correct, arr_energy_correct2, title, filepath, plot_range=(3, 100), ):
    """
    Makes a mpl step plot with Energy vs. Accuracy based on a [Energy, correct] array.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, ptype, is_cc, y_pred].
    :param str title: Title of the mpl step plot.
    :param str filepath: Filepath of the resulting plot.
    :param (int, int) plot_range: Plot range that should be used in the step plot. E.g. (3, 100) for 3-100GeV Data.
    """
    # Calculate accuracy in energy range
    energy = arr_energy_correct[:, 0]
    correct = arr_energy_correct[:, 1]

    hist_1d_energy = np.histogram(energy, bins=98, range=plot_range) #häufigkeit von energien
    hist_1d_energy_correct = np.histogram(arr_energy_correct[correct == 1, 0], bins=98, range=plot_range) #häufigkeit von richtigen energien
    hist_1d_energy_accuracy_bins = np.divide(hist_1d_energy_correct[0], hist_1d_energy[0], dtype=np.float32) #rel häufigkeit von richtigen energien
    
    #2
    energy2 = arr_energy_correct2[:, 0]
    correct2 = arr_energy_correct2[:, 1]

    hist_1d_energy2 = np.histogram(energy2, bins=98, range=plot_range) #häufigkeit von energien
    hist_1d_energy_correct2 = np.histogram(arr_energy_correct2[correct2 == 1, 0], bins=98, range=plot_range) #häufigkeit von richtigen energien
    hist_1d_energy_accuracy_bins2 = np.divide(hist_1d_energy_correct2[0], hist_1d_energy2[0], dtype=np.float32) #rel häufigkeit von richtigen energien
    
    # For making it work with matplotlib step plot
    bin_edges = hist_1d_energy[1]
    bin_edges_centered = bin_edges[:-1] + 0.5

    plt.step(bin_edges_centered, hist_1d_energy_accuracy_bins, where='mid', label="VGG")
    plt.step(bin_edges_centered, hist_1d_energy_accuracy_bins2, where='mid', label="Encoder")
    
    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.legend()
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Accuracy')
    plt.ylim((0.5, 1))
    plt.title(title)
    plt.grid(True)

    plt.savefig(filepath+"_comp.pdf")
    return(bin_edges_centered, hist_1d_energy_accuracy_bins, hist_1d_energy_accuracy_bins2)
    
    
    
def make_binned_data_plot(hist_data_array, label_array, title, y_label="Accuracy", y_lims=(0.5,1), color_array=[], legend_loc="best"):
    """
    Makes a plot based on multiple binned acc or loss data.
    Will plot for every hist data in the array: [1] over [0]
    """
    fig, ax = plt.subplots()
    for i, hist in enumerate(hist_data_array):
        #Use user defined colors, if given in proper length; else default palette
        energy=hist_data_array[i][0]
        y_axis_data=hist_data_array[i][1]
        
        if len(color_array) == len(hist_data_array):
            plt.step(energy, y_axis_data, where='post', label=label_array[i], color=color_array[i])
        else:
            plt.step(energy, y_axis_data, where='post', label=label_array[i])
            
    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.legend(loc=legend_loc)
    plt.xlabel('Energy [GeV]')
    plt.ylabel(y_label)
    plt.ylim((y_lims[0], y_lims[1]))
    plt.title(title)
    plt.grid(True)

    return fig
 
    

def make_energy_to_accuracy_plot_multiple_classes(arr_energy_correct_classes, title, filename, plot_range=(3,100)):
    """
    Makes a mpl step plot of Energy vs. 'Fraction of events classified as track' for multiple classes.
    Till now only used for muon-CC vs elec-CC.
    :param ndarray arr_energy_correct_classes: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param str title: Title that should be used in the plot.
    :param str filename: Filename that should be used for saving the plot.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    """
    fig, axes = plt.subplots()

    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}

    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'muon-CC', plot_range, linestyle='-', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'a_muon-CC', plot_range, linestyle='--', color='b')
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'elec-CC', plot_range, linestyle='-', color='r', invert=True)
    make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, 'a_elec-CC', plot_range, linestyle='--', color='r', invert=True)

    axes.legend(loc='center right')

    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.xlabel('Energy [GeV]')
    #plt.ylabel('Accuracy')
    plt.ylabel('Fraction of events classified as track')
    plt.ylim((0, 1.05))
    title = plt.title(title)
    title.set_position([.5, 1.04])
    plt.grid(True, zorder=0)

    plt.savefig(filename + '_3-100GeV.pdf')

    x_ticks_major = np.arange(0, 101, 5)
    plt.xticks(x_ticks_major)
    plt.xlim((0,40))
    plt.savefig(filename + '_3-40GeV.pdf')


def make_step_plot_1d_energy_accuracy_class(arr_energy_correct_classes, axes, particle_types_dict, particle_type, plot_range=(3,100), linestyle='-', color='b', invert=False):
    """
    Makes a mpl 1D step plot with Energy vs. Accuracy for a certain input class (e.g. a_muon-CC).
    :param ndarray arr_energy_correct_classes: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param mpl.axes axes: mpl axes object that refers to an existing plt.sublots object.
    :param dict particle_types_dict: Dictionary that contains a (particle_type, is_cc) [-> muon-CC!] tuple in order to classify the events.
    :param str particle_type: Particle type that should be plotted, e.g. 'a_muon-CC'.
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param str linestyle: Specifies the mpl linestyle that should be used.
    :param str color: Specifies the mpl color that should be used for plotting the step.
    :param bool invert: If True, it inverts the y-axis which may be useful for plotting a 'Fraction of events classified as track' plot.
    """
    class_vector = particle_types_dict[particle_type]

    arr_energy_correct_class = select_class(arr_energy_correct_classes, class_vector=class_vector)
    energy_class = arr_energy_correct_class[:, 0]
    correct_class = arr_energy_correct_class[:, 1]

    hist_1d_energy_class = np.histogram(energy_class, bins=98, range=plot_range)
    hist_1d_energy_correct_class = np.histogram(arr_energy_correct_class[correct_class == 1, 0], bins=98, range=plot_range)

    bin_edges = hist_1d_energy_class[1]
    hist_1d_energy_accuracy_class_bins = np.divide(hist_1d_energy_correct_class[0], hist_1d_energy_class[0], dtype=np.float32) # TODO solve division by zero

    if invert is True: hist_1d_energy_accuracy_class_bins = np.absolute(hist_1d_energy_accuracy_class_bins - 1)

    # For making it work with matplotlib step plot
    hist_1d_energy_accuracy_class_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_class_bins))

    axes.step(bin_edges, hist_1d_energy_accuracy_class_bins_leading_zero, where='pre', linestyle=linestyle, color=color, label=particle_type, zorder=3)


def select_class(arr_energy_correct_classes, class_vector):
    """
    Selects the rows in an arr_energy_correct_classes array that correspond to a certain class_vector.
    :param arr_energy_correct_classes: Array that contains the energy, correct, particle_type, is_cc [and y_pred] info for each event.
    :param (int, int) class_vector: Specifies the class that is used for filtering the array. E.g. (14,1) for muon-CC.
    """
    check_arr_for_class = arr_energy_correct_classes[:,2:4] == class_vector  # returns a bool for each of the class_vector entries

    # Select only the events, where every bool for one event is True
    indices_rows_with_class = np.logical_and(check_arr_for_class[:, 0], check_arr_for_class[:, 1])

    selected_rows_of_class = arr_energy_correct_classes[indices_rows_with_class]

    return selected_rows_of_class

#------------- Code for evaluation dataset script --------------------
    
def get_name_of_dump_files_for_evaluation_dataset(modelname, dataset, bins, class_type):
    dump_path = "/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/"
    #Give a seperate name for different class types
    if class_type is None:
        class_string = "_autoencoder"
    elif class_type[1]=="up_down":
        #for backward compatibility
        class_string = ""
    else:
        #e.g. energy, ...
        class_string = "_"+class_type[1]
        
    name_of_file= dump_path + modelname + "_" + dataset + class_string + "_"+str(bins)+"_bins_hist_data.txt"
    return name_of_file



#TODO: not bugfixed
def make_or_load_files(modelidents, dataset_array, bins, class_type=None):
    """
    Takes a bunch of models and returns the hist data for plotting, either
    by loading if it exists already or by generating it from scratch.
    Can also evaluate the performance of an autoencoder.
        
    Input:
        modelidents:    List of strs of the path to the models on which the evaluation is done on.
        dataset_array:  List of dataset tags on which the models will be evaluated on.
        bins:           Number of bins the evaluation will be binned to.
        class_type:     Class type of the prediction. None for autoencoders.
    Output:
        hist_data_array: A list of the evaluation for every model, that is: the binned data that
                         is used for making the plot (can contain binned acc,
                         mse, mre for track/shower ... )
    """
    #Extract the names of the models from their paths
    modelnames=[] # a tuple of eg       "vgg_1_xzt_supervised_up_down_epoch6" 
    #           (created from   "trained_vgg_1_xzt_supervised_up_down_epoch6.h5"   )
    for modelident in modelidents:
        modelnames.append(modelident.split("trained_")[1][:-3])
        
    hist_data_array=[]
    for i,modelname in enumerate(modelnames):
        dataset=dataset_array[i]
        print("Working on ",modelname,"using dataset", dataset, "with", bins, "bins")
        #Name of the dump file
        name_of_file=get_name_of_dump_files_for_evaluation_dataset(modelname, 
                                                       dataset, bins, class_type)
        
        if os.path.isfile(name_of_file)==True:
            #File was created before, just open and load
            hist_data_array.append(open_hist_data(name_of_file))
        else:
            #File has not been created before, generate new one
            hist_data = make_and_save_hist_data(dataset, modelidents[i], class_type, name_of_file, bins)
            hist_data_array.append(hist_data)
        print("Done.")
    return hist_data_array


#open dumped histogramm data, that was generated from the below functions
def open_hist_data(name_of_file):
    #hist data is list len 2 with 0:energy array, 1:acc/loss array
    print("Opening existing hist_data file", name_of_file)
    #load again
    with open(name_of_file, "rb") as dump_file:
        hist_data = pickle.load(dump_file)
    return hist_data



def make_and_save_hist_data(dataset, modelident, class_type, name_of_file, bins):
    """
    Calculate the evaluation of a model on a dataset, and then bin it energy wise.
    It is dumped automatically into the
    results/data folder, so that it has not to be generated again.
    """
    #Load necessary data
    model = load_model(modelident)
    dataset_info_dict = get_dataset_info(dataset)
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"]
    train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=32, n_bins=n_bins, n_gpu=1)

    print("Making energy_correct_array of ", modelident)
    
    if class_type is None:
        #This is for Autoencoders. Take MSE of original image vs predicted image
        
        #arr_energy_correct: [energy, Mean Squared error between original image and reconstructed one]
        arr_energy_correct = make_loss_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, broken_simulations_mode=broken_simulations_mode, swap_4d_channels=None, samples=None, dataset_info_dict=dataset_info_dict)
        #hist_data = [bin_edges_centered, hist_1d_energy_bins]:
        hist_data = make_energy_to_loss_data(arr_energy_correct, plot_range=(3,100), bins=bins)
    
    elif class_type == (1, "energy"):
        #This is for energy encoders. Take median relative error.
        swap_4d_channels = None
        #arr_energy_correct: [mc_energy, reco_energy, particle_type, is_cc]
        arr_energy_correct = make_performance_array_energy_energy(model, test_file, class_type, xs_mean, swap_4d_channels, dataset_info_dict, samples=None)
        # list len 2 of [energy_bins, hist_energy_losses, hist_energy_variance] 
        # for track and shower events
        hist_data = calculate_energy_mae_plot_data(arr_energy_correct)
        
    else:
        #For encoders with accuracy. Calculate accuracy.
        
        #arr_energy_correct: [energy, correct, particle_type, is_cc, y_pred] for every event
        arr_energy_correct = make_performance_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, broken_simulations_mode=broken_simulations_mode, swap_4d_channels=None, samples=None, dataset_info_dict=dataset_info_dict)
        #hist_data = [bin_edges_centered, hist_1d_energy_accuracy_bins]:
        hist_data = make_energy_to_accuracy_data(arr_energy_correct, plot_range=(3,100), bins=bins)
    
    print("Saving hist_data as", name_of_file)
    with open(name_of_file, "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data



# ------------- Functions used for energy-energy evaluation -------------#

def make_performance_array_energy_energy(model, f, class_type, xs_mean, swap_4d_channels, dataset_info_dict, samples=None):
    """
    Use a model to predict the energy reco on a dataset.
    :param ks.model.Model/Sequential model: Fully trained Keras model of a neural network.
    :param str f: Filepath of the file that is used for making predctions.
    :param (int, str) class_type: The number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param ndarray xs_mean: mean_image of the x dataset if zero-centering is enabled.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped in the generator.
    :param None/int samples: Number of events that should be predicted. If samples=None, the whole file will be used.
    :return: ndarray arr_energy_correct: Array that contains the mc_energy, reco_energy, particle_type, is_cc for each event.
    """
    # TODO only works for a single test_file till now
    n_bins = dataset_info_dict["n_bins"]
    batchsize = dataset_info_dict["batchsize"]
    broken_simulations_mode = dataset_info_dict["broken_simulations_mode"]
    
    generator = generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, zero_center_image=xs_mean, f_size=None , is_autoencoder=False, yield_mc_info=True, swap_col=swap_4d_channels, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict) # f_size=samples prob not necessary
    
    if samples is None: samples = len(h5py.File(f, 'r')['y'])
    steps = samples/batchsize


    #Output is for every event: [mc_energy, reco_energy, particle_type, is_cc]
    arr_energy_correct=None
    
    for s in range(int(steps)):
        if s % 600 == 0:
            print ('Predicting in step ' + str(s) + "/" + str(int(steps)))
        xs, y_true, mc_info = next(generator)
        reco_energy = model.predict_on_batch(xs) # shape (batchsize,1)
        reco_energy = np.reshape(reco_energy, reco_energy.shape[0]) #shape (batchsize,)
        
        #track info:
        #[event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, 
        # dir_x/y/z -> 5/6/7, time -> 8]
        
        mc_energy = mc_info[:, 2]
        particle_type = mc_info[:, 1]
        is_cc = mc_info[:, 3]

        ax = np.newaxis
        
        # make a temporary energy_correct array for this batch
        arr_energy_correct_temp = np.concatenate([mc_energy[:, ax], reco_energy[:, ax], particle_type[:, ax], is_cc[:, ax]], axis=1)

        if arr_energy_correct is None:
            arr_energy_correct = np.zeros((int(steps) * batchsize, arr_energy_correct_temp.shape[1:2][0]), dtype=np.float32)
        arr_energy_correct[s*batchsize : (s+1) * batchsize] = arr_energy_correct_temp


    print("\nStatistics of this reconstruction:")
    mc_energy = arr_energy_correct[:,0]
    reco_energy = arr_energy_correct[:,1]
    abs_err = np.abs(mc_energy - reco_energy)
    total_abs_mean = abs_err.mean()
    total_relative_median = np.median(abs_err/mc_energy)
    total_relative_variance = np.var(abs_err/mc_energy)
    print("Average mean absolute error over all energies:", total_abs_mean)
    print("Median relative error over all energies:",total_relative_median)
    print("Variance in relative error over all energies:", total_relative_variance)
    print(total_abs_mean,total_relative_median,total_relative_variance, "\n")
    
    
    return arr_energy_correct

def setup_and_make_energy_arr_energy_correct(model_path, dataset_info_dict, zero_center, samples=None):   
    """
    Comfort function to setup everything that is needed for generating the arr_energy_correct 
    for energy evaluation, and then generates it.
    """
    #The model that does the prediction
    model=load_model(model_path)
    #The dataset to be used to predict on; the prediction is done on the test file
    #home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    #broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
    filesize_factor=dataset_info_dict["filesize_factor"]
    #filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
        
    #Zero-Center with precalculated mean image
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    #test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    n_gpu=(1, 'avolkov')
    if zero_center == True:
        xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=n_gpu[0])
    else:
        xs_mean = None
        
    arr_energy_correct = make_performance_array_energy_energy(model, test_file, 
                        class_type=[1,"energy"], xs_mean=xs_mean, swap_4d_channels=None, 
                        dataset_info_dict=dataset_info_dict, samples=samples)
    return arr_energy_correct
    


def calculate_2d_hist_data(arr_energy_correct, energy_bins=np.arange(3,101,1)):
    """
    Take a list of [mc_energy, reco_energy, particle_type, is_cc] for many events
    and generate 2d numpy histograms for track/shower events from it.
    """  
    mc_energy = arr_energy_correct[:,0]
    reco_energy = arr_energy_correct[:,1]
    is_track, is_shower = track_shower_seperation(arr_energy_correct[:,2], arr_energy_correct[:,3])
    
    hist_2d_data_track = np.histogram2d(mc_energy[is_track], reco_energy[is_track], energy_bins)
    hist_2d_data_shower = np.histogram2d(mc_energy[is_shower], reco_energy[is_shower], energy_bins)
    
    hist_2d_data = [hist_2d_data_track, hist_2d_data_shower]
    return hist_2d_data


def track_shower_seperation(particle_type, is_cc):
    #Input: np arrays from mc_info
    #Output: np arrays type bool which identify track/shower like events
    
    #track: muon/a-muon CC --> 14, True
    #shower: elec/a-elec; muon/a-muon NC
    #particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
    # In my dataset, there are actually no NC events
    
    abs_particle_type = np.abs(particle_type)
    track  = np.logical_and(abs_particle_type == 14, is_cc==True)
    shower = np.logical_or(np.logical_and(abs_particle_type == 14, is_cc==False), 
                           abs_particle_type==12)
    
    return track, shower

def norm_columns_of_2d_hist_data(z):
    """
    Takes the output [0] of np.hist2d and normalizes each column.
    Looks bad when plotted, so not used.
    """
    #z shape e.g. 97,97
    #z = z/np.sum(z,axis=1)
    for column_index in range(len(z)):
        z[column_index] = z[column_index]/np.sum(z[column_index])
    
    return z

def make_2d_hist_plot(hist_2d_data, seperate_track_shower=True, normalize_columns=False):
    """
    Takes a numpy 2d histogramm of mc-energy vs reco-energy and returns
    a plot.
    """
    hist_2d_data_track, hist_2d_data_shower = hist_2d_data
    #Bin edges are the same for both histograms
    x=hist_2d_data_track[1] #mc energy bin edges
    y=hist_2d_data_track[2] #reco energy bin edges
    
    title="Energy reconstruction"
    xlabel = "True energy (GeV)"
    ylabel = "Reconstructed energy (GeV)"
    
    if normalize_columns==False:
        cbar_label = 'Number of events'
    else:
        cbar_label = "Fraction of counts"
        
    if seperate_track_shower == False:
        #counts; this needs to be transposed to be displayed properly for reasons unbeknownst
        z=hist_2d_data_track[0].T + hist_2d_data_shower[0].T
        if normalize_columns == True:
            z=norm_columns_of_2d_hist_data(z)
        
        fig, ax = plt.subplots()
        plot = ax.pcolormesh(x,y,z, norm=colors.LogNorm(vmin=1, vmax=z.max()))
        
        fig.suptitle(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        cbar = plt.colorbar(plot)
        cbar.ax.set_ylabel(cbar_label)
        
    else:
        z1=hist_2d_data_track[0].T 
        z2=hist_2d_data_shower[0].T

        if normalize_columns == True:
            z1=norm_columns_of_2d_hist_data(z1)
            z2=norm_columns_of_2d_hist_data(z2)

        fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12,4.8))
        plot1 = ax1.pcolormesh(x,y,z1, norm=colors.LogNorm(vmin=1, vmax=z1.max()))
        ax1.set_title("Track like events")
        cbar1 = fig.colorbar(plot1, ax=ax1, )
        cbar1.ax.set_ylabel(cbar_label)
        
        plot2 = ax2.pcolormesh(x,y,z2, norm=colors.LogNorm(vmin=1, vmax=z2.max()))
        ax2.set_title("Shower like events")
        cbar2 = fig.colorbar(plot2, ax=ax2)
        cbar2.ax.set_ylabel(cbar_label)
        
        
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_aspect("equal")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_aspect("equal")
        #plt.tight_layout(pad=2)
        plt.subplots_adjust(top=0.85, left=0.05, right=0.97)
        fig.suptitle(title, fontsize=18, y=0.98)
        
    return(fig)

def calculate_energy_mae_plot_data(arr_energy_correct, energy_bins=np.arange(3,101,1)):
    """
    Generate binned statistics for the energy mae, or the relative mae.
    seperate for track and shower events.
    """
    mc_energy = arr_energy_correct[:,0]
    reco_energy = arr_energy_correct[:,1]
    is_track, is_shower = track_shower_seperation(arr_energy_correct[:,2], arr_energy_correct[:,3])
    
    abs_err = np.abs(mc_energy - reco_energy)

    def bin_abs_error(energy_bins, mc_energy, abs_err, operation="median_relative"):
        #bin the abs_err, depending on their mc_energy, into energy_bins
        hist_energy_losses=np.zeros((len(energy_bins)-1))
        hist_energy_variance=np.zeros((len(energy_bins)-1))
        #In which bin does each event belong, according to its mc energy:
        bin_indices = np.digitize(mc_energy, bins=energy_bins)
        
        #For every mc energy bin, mean over the mae of all events that have a corresponding mc energy
        for bin_no in range(min(bin_indices), max(bin_indices)+1):
            current_abs_err = abs_err[bin_indices==bin_no]
            current_mc_energy = mc_energy[bin_indices==bin_no]
            
            if operation=="mae":
                #calculate mean absolute error (outdated)
                hist_energy_losses[bin_no-1]   = np.mean(current_abs_err)
            elif operation=="median_relative":
                #calculate the median of the relative error: |E_true-E_reco|/E_true
                #and also its variance
                relative_error = current_abs_err/current_mc_energy
                hist_energy_losses[bin_no-1]   = np.median(relative_error)
                hist_energy_variance[bin_no-1] = np.var(relative_error)
        #For proper plotting with plt.step where="post"
        hist_energy_losses=np.append(hist_energy_losses, hist_energy_losses[-1])
        hist_energy_variance=np.append(hist_energy_variance, hist_energy_variance[-1])
        energy_mae_plot_data = [energy_bins, hist_energy_losses, hist_energy_variance]
        return energy_mae_plot_data
    
    energy_mae_plot_data_track = bin_abs_error(energy_bins, mc_energy[is_track], abs_err[is_track])
    energy_mae_plot_data_shower = bin_abs_error(energy_bins, mc_energy[is_shower], abs_err[is_shower])
    energy_mae_plot_data = [energy_mae_plot_data_track, energy_mae_plot_data_shower]
    return energy_mae_plot_data


def make_energy_mae_plot(energy_mae_plot_data_list, seperate_track_shower=True, label_list=[]):
    """
    Takes a list of mae_plot_data (e.g. for different models) and makes a single plot.
    """
    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12,4.8))
    legend_handles = []
    for i,energy_mae_plot_data in enumerate(energy_mae_plot_data_list):
        energy_mae_plot_data_track, energy_mae_plot_data_shower = energy_mae_plot_data
        bins = energy_mae_plot_data_track[0]
        mean_track  = energy_mae_plot_data_track[1]
        mean_shower = energy_mae_plot_data_shower[1]
        var_track  = energy_mae_plot_data_track[2]
        var_shower = energy_mae_plot_data_shower[2]
        
        try:
            label = label_list[i]
        except IndexError:
            label = "unknown"
            
        if seperate_track_shower == False:
            summed_error=mean_track+mean_shower
            summed_var = var_shower+var_track
            plot1 = ax1.step(bins, summed_error, where='post')
            color_of_this_model = plot1[0].get_color()
            ax2.step(bins, summed_var, where='post', color=color_of_this_model)
            
            #Get an entry for the legend
            legend_entry = mpatches.Patch(color=color_of_this_model, label=label)
            legend_handles.append(legend_entry)
        else:
            #Plot the mean in left plot
            shower = ax1.step(bins, mean_shower, linestyle="--", where='post')
            color_of_this_model = shower[0].get_color()
            ax1.step(bins, mean_track, linestyle="-", where='post', color=color_of_this_model)
            
            #Plot the variance in right plot
            ax2.step(bins, var_shower, linestyle="--", where='post', color=color_of_this_model)
            ax2.step(bins, var_track, linestyle="-", where='post', color=color_of_this_model)
            
            #Get an entry for the legend
            legend_entry = mpatches.Patch(color=color_of_this_model, label=label)
            legend_handles.append(legend_entry)
        
    x_ticks_major = np.arange(0, 101, 10)
    ax1.set_xticks(x_ticks_major)
    ax1.minorticks_on()
    ax2.set_xticks(x_ticks_major)
    ax2.minorticks_on()
    
    ax1.set_xlabel('True energy (GeV)')
    ax1.set_ylabel('Median fractional energy resolution')
    
    ax2.set_xlabel('True energy (GeV)')
    ax2.set_ylabel(r'Variance of relative Error (GeV$^2$)')
    
    #plt.ylim((0, 0.2))
    fig.suptitle("Energy reconstruction", fontsize=16)
    ax1.grid(True)
    ax2.grid(True)
    
    #Get control over the legend entries
    #legend1 = ax2.legend(handles=legend_handles, loc="upper right")
    #ax2.add_artist(legend1)
    
    #Make a second legend box
    track_line  = mlines.Line2D([], [], color='grey', linestyle="-",  label='Track')
    shower_line = mlines.Line2D([], [], color='grey', linestyle="--", label='Shower')
    empty_line = mlines.Line2D([], [], color='white', linestyle="", label='')
    legend2_handles = [track_line,shower_line,empty_line]
    #legend2 = ax2.legend(handles=legend2_handles, loc="best")
    #ax2.add_artist(legend2)
   
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=legend2_handles+legend_handles)
    
    plt.subplots_adjust(top=0.85, left=0.06, right=0.85)
    
    return fig

def test_energy_evaluation_functions(mode="real"):
    """
    Generate some random noise and plot it with the energy evaluation functions.
    """ 
    if mode=="noise":
        how_many = 50000
        dummy_hits_1 = np.random.rand(how_many,1)*100
        dummy_hits_2 = np.random.rand(how_many,1)*100
        #dummy_hits_1 = np.logspace(0,100,how_many) 
        #dummy_hits_1 = np.reshape(dummy_hits_1, dummy_hits_1.shape+(1,))
        #dummy_hits_2 = np.logspace(0,100,how_many)+np.random.rand(how_many)*10
        #dummy_hits_2 = np.reshape(dummy_hits_2, dummy_hits_2.shape+(1,))
        
        dummy_types = np.ones((how_many,1))*12 + np.random.randint(0,2,size=(how_many,1))*2
        dummy_cc = np.ones((how_many,1))
        dummy_input = np.concatenate([dummy_hits_1, dummy_hits_2, dummy_types, dummy_cc], axis=-1)
        
        #make_2d_hist_plot(calculate_2d_hist_data(dummy_input), seperate_track_shower=0, normalize_columns=0)
        make_energy_mae_plot([calculate_energy_mae_plot_data(dummy_input),])
    elif mode=="real":
        data2d = np.load("temp.npy")
        make_2d_hist_plot(data2d)
        make_energy_mae_plot(data2d)


# ------------- Functions used in making Matplotlib plots -------------#


def make_prob_hists(arr_energy_correct, modelname):
    """
    Function that makes (class-) probability histograms based on the arr_energy_correct.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, energy, ptype, is_cc, y_pred].
    :param str modelname: Name of the model that is used for saving the plots.
    """
    def configure_hstack_plot(plot_title, savepath):
        """
        Configure a mpl plot with GridLines, Logscale etc.
        :param str plot_title: Title that should be used for the plot.
        :param str savepath: path that should be used for saving the plot.
        """
        axes.legend(loc='upper center')
        plt.grid(True, zorder=0)
        #plt.yscale('log')

        x_ticks_major = np.arange(0, 1.1, 0.1)
        plt.xticks(x_ticks_major)
        plt.minorticks_on()

        plt.xlabel('Probability')
        plt.ylabel('Normed Quantity')
        title = plt.title(plot_title)
        title.set_position([.5, 1.04])

        plt.savefig(savepath)

    fig, axes = plt.subplots()
    particle_types_dict = {'muon-CC': (14, 1), 'a_muon-CC': (-14, 1), 'elec-CC': (12, 1), 'a_elec-CC': (-12, 1)}

    # make energy cut, 3-40GeV
    arr_energy_correct_ecut = arr_energy_correct[arr_energy_correct[:, 0] <= 40]

    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'muon-CC', 0, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_muon-CC', 0, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'elec-CC', 0, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_elec-CC', 0, plot_range=(0, 1), color='r', linestyle='--')

    configure_hstack_plot(plot_title='Probability to be classified as elec-CC (shower)', savepath='results/plots/PT_hist1D_prob_shower_' + modelname + '.pdf')
    plt.cla()

    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'muon-CC', 1, plot_range=(0,1), color='b', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_muon-CC', 1, plot_range=(0, 1), color='b', linestyle='--')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'elec-CC', 1, plot_range=(0,1), color='r', linestyle='-')
    make_prob_hist_class(arr_energy_correct_ecut, axes, particle_types_dict, 'a_elec-CC', 1, plot_range=(0, 1), color='r', linestyle='--')

    configure_hstack_plot(plot_title='Probability to be classified as muon-CC (track)', savepath='results/plots/PT_hist1D_prob_track_' + modelname + '.pdf')
    plt.cla()


def make_prob_hist_class(arr_energy_correct, axes, particle_types_dict, particle_type, prob_class_index, plot_range=(0,1), color='b', linestyle='-'):
    """
    Makes mpl hists based on an arr_energy_correct for a certain particle class (e.g. 'muon-CC').
    :param ndarray arr_energy_correct: Array that contains the energy, correct, particle_type, is_cc and y_pred info for each event.
    :param mpl.axes axes: mpl axes object that refers to an existing plt.sublots object.
    :param dict particle_types_dict: Dictionary that contains a (particle_type, is_cc) [-> muon-CC!] tuple in order to classify the events.
    :param str particle_type: Particle type that should be plotted, e.g. 'a_muon-CC'.
    :param int prob_class_index: Specifies which class (e.g. elec-CC/muon-CC) should be used for the probability plots.
                                 E.g. for 2 classes: [1,0] -> shower -> index=0, [0,1] -> track -> index=1
    :param (int, int) plot_range: Tuple that specifies the X-Range of the plot.
    :param str color: Specifies the mpl color that should be used for plotting the hist.
    :param str linestyle: Specifies the mpl linestyle that should be used.
    """
    ptype_vector = particle_types_dict[particle_type]
    arr_energy_correct_ptype = select_class(arr_energy_correct, class_vector=ptype_vector)

    # prob_class_index = 0/1, 0 is shower, 1 is track
    prob_ptype_class = arr_energy_correct_ptype[:, 4 + prob_class_index]

    hist_1d_prob_ptype_class = axes.hist(prob_ptype_class, bins=40, range=plot_range, normed=True, color=color, label=particle_type, histtype='step', linestyle=linestyle, zorder=3)






