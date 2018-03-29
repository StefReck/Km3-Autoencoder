# -*- coding: utf-8 -*-
"""
Load a channel id autoencoder model and predict on some train files, then plot it optionally.
"""

from keras.models import load_model
from keras import metrics
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt

from get_dataset_info import get_dataset_info
from util.run_cnn import load_zero_center_data, generate_batches_from_hdf5_file, h5_get_number_of_rows

def parse_input():
    parser = argparse.ArgumentParser(description='Predict on channel data')
    parser.add_argument('model_name', type=str)

    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
model_name = params["model_name"]
mode="statistics"
zero_center = False

#for plot mode, number of 32 batches of channel_id arrays should be read through for the plot
how_many_dom_batches = 1000
bins=100

model=load_model(model_name)
dataset_info_dict=get_dataset_info("xyzc_flat")

if mode == "simple":
    #Print some 31-arrays and the prediction from the autoencoder
    how_many_doms=10 #to read from file
    minimum_counts = 5
    
    test_file = dataset_info_dict["test_file"]
    
    if zero_center==True:
        xs_mean=load_zero_center_data(((dataset_info_dict["train_file"],),), batchsize=32, n_bins=dataset_info_dict["n_bins"], n_gpu=1)
    else:
        xs_mean = 0
    
    f = h5py.File(test_file, "r")
    
    #look for some doms that are not mostly 0
    batch=[]
    i=0
    while len(batch)<=how_many_doms:
        dom=f["x"][i:i+1]
        if dom.sum()>=minimum_counts:
            batch.extend(dom)
        i+=1
        
    batch=np.array(batch) 
    batch_centered=np.subtract(batch, xs_mean)
    pred=np.add(model.predict_on_batch(batch_centered), xs_mean)
    
    for i in range(len(batch)):
        print("Original")
        print(batch[i])
        print("Prediction")
        print(pred[i])
        print("loss:", ((batch[i]-pred[i])**2).mean())
        print("\n")

elif mode=="plot":
    #make plot of predictions
    #measured counts are almost always 0 or 1
    maximum_counts_to_look_for=1
    skip_zero_counts=False
    
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
    filesize_factor=dataset_info_dict["filesize_factor"]
    filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
    
    print("Total channel ids:", how_many_dom_batches*batchsize*31)
    
    class_type=(2,"up_down")
    is_autoencoder=True
    
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    #test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    
    if zero_center==True:
        xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=1)
    else:
        xs_mean=0
    
    generator = generate_batches_from_hdf5_file(test_file, batchsize, n_bins, class_type, 
                                    is_autoencoder, dataset_info_dict, broken_simulations_mode=0,
                                    f_size=None, zero_center_image=xs_mean, yield_mc_info=False,
                                    swap_col=None, is_in_test_mode = False)
    
    #prediction on channel id that measured
    #          0 ,1 ,2 ,3 ...  counts
    pred_on = []
    
    for measured_counts in range(maximum_counts_to_look_for+1):
        pred_on.append([])
    
    print_something_after_every = int(how_many_dom_batches/10)
    for i in range(how_many_dom_batches):
        if i%print_something_after_every==0:
            print("Predicting ... ", int(10*i/print_something_after_every), "% done")
            
        data=next(generator)[0]
        data_real = np.add(data, xs_mean)
        pred=np.add(model.predict_on_batch(data), xs_mean)
        
        #data_real is still a batch of len batchsize of single doms (dim. e.g. (32,31)), so look at each one:
        for dom_no,data_real_single in enumerate(data_real):
            pred_single=pred[dom_no]
            for measured_counts in range(skip_zero_counts, maximum_counts_to_look_for+1):
                #sort predicitions into list according to original counts
                pred_on[measured_counts].extend(pred_single[data_real_single==measured_counts])
                
    print("Done, generating plot...")
    
    plt.title("Channel autoencoder predictions (%)")
    plt.ylabel("Fraction of predicitons")
    plt.xlabel("Predicted counts")
    plt.plot([],[], " ", label="Original counts")
    
    make_plots_of_counts=list(range(maximum_counts_to_look_for+1))
    
    ex_list=[]
    #fill with maximum and minimum prediction of every original count number
    for counts_array in pred_on:
        len(counts_array)
        if len(counts_array) != 0:
            ex_list.extend([np.amax(counts_array), np.amin(counts_array)])
    range_of_plot=[np.amin(ex_list),np.amax(ex_list)]

    #relative width of bins as fracton of bin size
    #relative_width=1/len(make_plots_of_counts)
    #bin_size = (range_of_plot[0]-range_of_plot[1]) / bins
    #bin_edges = np.linspace(range_of_plot[0], range_of_plot[1], num=bins+1)
    
    for c in make_plots_of_counts:
        if len(pred[c]) != 0:
            #offset = bin_size*relative_width*c
            plt.hist( x=pred_on[c], bins=bins, label=str(c), density=True, range=range_of_plot )
    plt.legend()
    plt.show()
        
    
elif mode=="statistics":
    #evaluate on test set. Check wheter doms with n hits in total were reconstructed correctly.
    #For this, predictions are rounded to next integer
    
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
    filesize_factor=dataset_info_dict["filesize_factor"]
    filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    
    #higher for testing
    batchsize=32
    dataset_info_dict["batchsize"]=batchsize #def 32
    

    class_type=(2,"up_down")
    is_autoencoder=True
    
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    
    if zero_center==True:
        xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=1)
    else:
        xs_mean=None
    
    generator = generate_batches_from_hdf5_file(test_file, batchsize, n_bins, class_type, 
                                    is_autoencoder, dataset_info_dict, broken_simulations_mode=0,
                                    f_size=None, zero_center_image=xs_mean, yield_mc_info=False,
                                    swap_col=None, is_in_test_mode = False)
    
    total_number_of_batches = int(test_tuple[0][1]/batchsize)
    total_number_of_batches=10
    print("Filesize:",test_tuple[0][1], "Total number of batches:", total_number_of_batches)
    
    
    #a dict with entries: total_counts_per_dom : [[correct_from_batch_0, ...],[total_from_batch_0,...]]
    #e.g.                 0 : [[70,75,...],[96,94,...]]
    counts_dict={}
    
    for batchno in range(total_number_of_batches):
        data = next(generator)[0]
        print("data shape", data.shape)
        prediction = np.round(model.predict_on_batch(data))
        
        #shape (batchsize,)
        total_counts_measured_in_dom = np.sum(data, axis=1)
        print("total_counts shape",total_counts_measured_in_dom.shape)
        
        #Should result in a (batchsize,) array that states wheter the whole dom was predicted correctly
        dom_correct = np.logical_and.reduce(data==prediction, axis=1)
        print("dom_correct shape",dom_correct.shape)
        
        #which count numbers were measured in all the doms
        counts=np.unique(total_counts_measured_in_dom).astype(int)
        
        for count in counts:
            positions = np.where(data==count)
            predicted_correct_there = np.sum(dom_correct[positions]).astype(int)
            total_doms_with_these_counts = len(positions)
            
            if count not in counts_dict:
                counts_dict[count]=[[],[]]
                
            counts_dict[count][0].append(predicted_correct_there)
            counts_dict[count][1].append(total_doms_with_these_counts)
                
    print(counts_dict)
            
            