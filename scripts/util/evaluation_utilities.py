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
    
def make_energy_to_accuracy_plot_comp_data(hist_data_array, label_array, title, filepath, y_label="Accuracy", y_lims=(0.5,1), color_array=[], legend_loc="best"):
    """
    Makes a mpl step plot with Energy vs. Accuracy based on a [Energy, correct] array.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, ptype, is_cc, y_pred].
    :param str title: Title of the mpl step plot.
    :param str filepath: Filepath of the resulting plot.
    :param (int, int) plot_range: Plot range that should be used in the step plot. E.g. (3, 100) for 3-100GeV Data.
    """
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

    plt.savefig(filepath)
    plt.close()
    
def make_energy_to_loss_plot_comp_data(hist_data_array, label_array, title, filepath, y_label="Loss"):
    """
    Makes a mpl step plot with Energy vs. Accuracy based on a [Energy, correct] array.
    :param ndarray(ndim=2) arr_energy_correct: 2D array with the content [Energy, correct, ptype, is_cc, y_pred].
    :param str title: Title of the mpl step plot.
    :param str filepath: Filepath of the resulting plot.
    :param (int, int) plot_range: Plot range that should be used in the step plot. E.g. (3, 100) for 3-100GeV Data.
    """
    for i, hist in enumerate(hist_data_array):
        plt.step(hist_data_array[i][0], hist_data_array[i][1], where='post', label=label_array[i])
    
    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.legend()
    plt.xlabel('Energy [GeV]')
    plt.ylabel(y_label)
    #plt.ylim((0, 0.2))
    plt.title(title)
    plt.grid(True)

    plt.savefig(filepath)

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

#------------- My Code --------------------

#Takes a bunch of models and returns the hist data for plotting, either
#by loading if it exists already or by generating it from scratch
#can also evaluate the performance of an autoencoder 
#TODO: not bugfixed
def make_or_load_files(modelnames, dataset_array, bins, modelidents=None, modelpath=None, class_type=None, is_autoencoder_list=None):
    if is_autoencoder_list==None:
        #default: no autoencoders
        is_autoencoder_list=np.zeros_like(modelnames)
        
    hist_data_array=[]
    for i,modelname in enumerate(modelnames):
        dataset=dataset_array[i]
        is_autoencoder = is_autoencoder_list[i]
        print("Working on ",modelname,"using dataset", dataset, "with", bins, "bins")
        
        name_of_file="/home/woody/capn/mppi013h/Km3-Autoencoder/results/data/" + modelname + "_" + dataset + "_"+str(bins)+"_bins_hist_data.txt"
        
        
        if os.path.isfile(name_of_file)==True:
            hist_data_array.append(open_hist_data(name_of_file))
        else:
            
            if is_autoencoder == 1:
                hist_data = make_and_save_hist_data_autoencoder(modelpath, dataset, modelidents[i], class_type, name_of_file, bins)
            else:
                hist_data = make_and_save_hist_data(modelpath, dataset, modelidents[i], class_type, name_of_file, bins)
            
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


#Accuracy as a function of energy binned to a histogramm. It is dumped automatically into the
#results/data folder, so that it has not to be generated again
def make_and_save_hist_data(modelpath, dataset, modelident, class_type, name_of_file, bins):
    model = load_model(modelpath + modelident)
    
    dataset_info_dict = get_dataset_info(dataset)
    #home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"]
    
    train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=32, n_bins=n_bins, n_gpu=1)

    
    print("Making energy_correct_array of ", modelident)
    arr_energy_correct = make_performance_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, broken_simulations_mode=broken_simulations_mode, swap_4d_channels=None, samples=None, dataset_info_dict=dataset_info_dict)
    #hist_data = [bin_edges_centered, hist_1d_energy_accuracy_bins]:
    hist_data = make_energy_to_accuracy_data(arr_energy_correct, plot_range=(3,100), bins=bins)
    #save to file
    print("Saving hist_data as", name_of_file)
    with open(name_of_file, "wb") as dump_file:
        pickle.dump(hist_data, dump_file)
    return hist_data



#Loss of an AE as a function of energy, rest like above
def make_and_save_hist_data_autoencoder(modelpath, dataset, modelident, class_type, name_of_file, bins):
    model = load_model(modelpath + modelident)
    
    dataset_info_dict = get_dataset_info(dataset)
    #home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"]
    
    train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=32, n_bins=n_bins, n_gpu=1)
    
    
    print("Making energy_correct_array of ", modelident)
    arr_energy_correct = make_loss_array_energy_correct(model=model, f=test_file, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, batchsize = 32, broken_simulations_mode=broken_simulations_mode, swap_4d_channels=None, samples=None, dataset_info_dict=dataset_info_dict)
    hist_data = make_energy_to_loss_data(arr_energy_correct, plot_range=(3,100), bins=bins)
    #save to file
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
        if s % 300 == 0:
            print ('Predicting in step ' + str(s) + "/" + str(int(steps)))
        xs, y_true, mc_info = next(generator)
        reco_energy = model.predict_on_batch(xs) # shape (batchsize,1)
        reco_energy = np.reshape(reco_energy, reco_energy.shape[0]) #shape (batchsize,)

        mc_energy = mc_info[:, 2]
        particle_type = mc_info[:, 1]
        is_cc = mc_info[:, 3]

        ax = np.newaxis
        
        # make a temporary energy_correct array for this batch
        arr_energy_correct_temp = np.concatenate([mc_energy[:, ax], reco_energy[:, ax], particle_type[:, ax], is_cc[:, ax]], axis=1)

        if arr_energy_correct is None:
            arr_energy_correct = np.zeros((int(steps) * batchsize, arr_energy_correct_temp.shape[1:2][0]), dtype=np.float32)
        arr_energy_correct[s*batchsize : (s+1) * batchsize] = arr_energy_correct_temp

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
    and generate a 2d numpy histogram from it.
    """  
    mc_energy = arr_energy_correct[:,0]
    reco_energy = arr_energy_correct[:,1]
    
    hist_2d_data = np.histogram2d(mc_energy, reco_energy, energy_bins)
    return hist_2d_data

def make_2d_hist_plot(hist_2d_data):
    """
    Takes a numpy 2d histogramm of mc-energy vs reco-energy and returns
    a plot.
    """
    z=hist_2d_data[0].T #counts; this needs to be transposed to be displayed properly for reasons unbeknownst
    x=hist_2d_data[1] #mc energy bin edges
    y=hist_2d_data[2] #reco energy bin edges
    
    
    fig, ax = plt.subplots()
    plot = ax.pcolormesh(x,y,z, norm=colors.LogNorm(vmin=1, vmax=z.max()))
    
    ax.set_title("Energy reconstruction")
    ax.set_xlabel("True energy (GeV)")
    ax.set_ylabel("Reconstructed energy (GeV)")
    
    cbar = plt.colorbar(plot)
    cbar.ax.set_ylabel('Number of events')
    
    return(fig)


def calculate_energy_mae_plot_data(arr_energy_correct, energy_bins=np.arange(3,101,1)):
    # Generate the data for a plot in which mc_energy vs mae is shown.
    mc_energy = arr_energy_correct[:,0]
    
    reco_energy = arr_energy_correct[:,1]
    abs_err = np.abs(mc_energy - reco_energy)

    hist_energy_losses=np.zeros((len(energy_bins)-1))
    #In which bin does each event belong, according to its mc energy:
    bin_indices = np.digitize(mc_energy, bins=energy_bins)
    #For every mc energy bin, mean over the mae of all events that have a corresponding mc energy
    for bin_no in range(min(bin_indices), max(bin_indices)+1):
        hist_energy_losses[bin_no-1] = np.mean(abs_err[bin_indices==bin_no])
    print("Average mean absolute error over all energies:", abs_err.mean())
    #For proper plotting with plt.step where="post"
    hist_energy_losses=np.append(hist_energy_losses, hist_energy_losses[-1])
    energy_mae_plot_data = [energy_bins, hist_energy_losses]
    return energy_mae_plot_data

def make_energy_mae_plot(energy_mae_plot_data):
    fig, ax = plt.subplots()
    plt.step(energy_mae_plot_data[0], energy_mae_plot_data[1], where='post')
    x_ticks_major = np.arange(0, 101, 10)
    plt.xticks(x_ticks_major)
    plt.minorticks_on()

    plt.legend()
    plt.xlabel('True energy (GeV)')
    plt.ylabel('Mean absolute error (GeV)')
    #plt.ylim((0, 0.2))
    plt.title("Energy reconstruction performance")
    plt.grid(True)

    return fig


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






