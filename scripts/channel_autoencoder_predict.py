# -*- coding: utf-8 -*-
"""
Load a channel id autoencoder model and predict on some train files.
"""

from keras.models import load_model
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

mode="plot"

model=load_model(model_name)
dataset_info_dict=get_dataset_info("xyzc_flat")

if mode == "simple":
    #Print some 31-arrays and the prediction from the autoencoder
    how_many_doms=10 #to read from file
    minimum_counts = 5
    
    test_file = dataset_info_dict["test_file"]
    xs_mean=load_zero_center_data(((dataset_info_dict["train_file"],),), batchsize=32, n_bins=dataset_info_dict["n_bins"], n_gpu=1)
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
    how_many_dom_batches = 100
    skip_zero_counts=True
    
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
    
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=1)

    
    generator = generate_batches_from_hdf5_file(test_file, batchsize, n_bins, class_type, 
                                    is_autoencoder, dataset_info_dict, broken_simulations_mode=0,
                                    f_size=None, zero_center_image=xs_mean, yield_mc_info=False,
                                    swap_col=None, is_in_test_mode = False)
    
    #prediction on channel id that measured
    #          0 ,1 ,2 ,3 ...  counts
    pred_on = []
    maximum_counts_to_look_for=3
    
    for measured_counts in range(maximum_counts_to_look_for+1):
        pred_on.append([])
        
    for i in range(how_many_dom_batches):
        data=next(generator)[0]
        
        data_real = np.add(data, xs_mean)
        pred=np.add(model.predict_on_batch(data), xs_mean)
        
        #data_real is still a batch of len batchsize of singe doms, so look at each one:
        for dom_no,data_real_single in enumerate(data_real):
            pred_single=pred[dom_no]
            for measured_counts in range(skip_zero_counts, maximum_counts_to_look_for+1):
                pred_on[measured_counts].extend(pred_single[data_real_single==measured_counts])
    
    make_plots_of_counts=[1,2,3]
    for c in make_plots_of_counts:
        if len(pred[c]) != 0:
            plt.hist( pred_on[c], label=str(c), bins=50 )
    plt.legend()
    plt.show()
        
    

