# -*- coding: utf-8 -*-
"""
Calculate the mse and the mse of bins above and below a threshold for some autoencoders.
"""
from keras.models import load_model
from os import fsync
import numpy as np

from util.run_cnn import load_zero_center_data, generate_batches_from_hdf5_file, h5_get_number_of_rows
from get_dataset_info import get_dataset_info
from util.custom_loss_functions import get_custom_objects

home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"

#array of strings that identifiy the models, up to the epoch
model_bases=[home_path+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_epoch",
             home_path+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_epoch"]
#name of logfiles (auto generated)
names_of_log_files = [home_path+"results/data/mse_"+model_bases[i].split("/")[-1][:-6]+"s.txt" for i in range(len(model_bases))]


#epochs to make datapoints for (if 1 is the first entry, the head line will be printed in the logfile)
epochs_of_model=np.arange(1,100)
#dataset to test on
dataset_tag="xzt"
#bins with more hits then this will have their own mse calculated
threshold_greater_then=3







def setup_generator_testfile(class_type, is_autoencoder, dataset_info_dict, yield_mc_info=False):
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    #broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
    filesize_factor=dataset_info_dict["filesize_factor"]
    #filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
    
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    #test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=1)
    
    generator = generate_batches_from_hdf5_file(test_file, batchsize, n_bins, class_type, 
                                    is_autoencoder, dataset_info_dict, broken_simulations_mode=0,
                                    f_size=None, zero_center_image=xs_mean, yield_mc_info=yield_mc_info,
                                    swap_col=None, is_in_test_mode = False)
    return generator

def make_logfile(name_of_log_file, epochs_of_model, model_base, dataset_tag, threshold_greater_then):
    batchsize_for_testing=128
    class_type=None #not needed for AE
    is_autoencoder=True
    dataset_info_dict = get_dataset_info(dataset_tag)
    dataset_info_dict["batchsize"] = batchsize_for_testing
    
    custom_objects=get_custom_objects()
    
    with open(name_of_log_file, 'a') as logfile:
        if epochs_of_model[0]==1:
            logfile.write("#Epoch\t"+"MSE above"+str(threshold_greater_then)+"\t"+"MSE below""\t"+"MSE\n")
        
        for epoch in epochs_of_model:
            print("Loading model:", model_base+str(epoch)+".h5")
            autoencoder = load_model(model_base+str(epoch)+".h5", custom_objects=custom_objects)
            generator = setup_generator_testfile(class_type, is_autoencoder, dataset_info_dict)
            
            test_file=dataset_info_dict["test_file"]
            filesize_factor_test=dataset_info_dict["filesize_factor_test"]
            file_size = int(h5_get_number_of_rows(test_file)*filesize_factor_test)
            
            mse_above_list=np.zeros(int(file_size/batchsize_for_testing))
            mse_below_list=np.zeros(int(file_size/batchsize_for_testing))
            mse_list=np.zeros(int(file_size/batchsize_for_testing))
            
            print("Starting testing on epoch",epoch)
            for batch_no in range(int(file_size/batchsize_for_testing)):
                data=next(generator)[0]
                prediction = autoencoder.predict_on_batch(data)
                se = ( (prediction-data)**2 )
                
                above_threshold=data>threshold_greater_then
                mse_above = se[ above_threshold            ].mean()
                mse_below = se[ np.invert(above_threshold) ].mean()
                mse = se.mean()
                
                mse_above_list[batch_no]=mse_above
                mse_below_list[batch_no]=mse_below
                mse_list[batch_no]=mse
            print("Completed.")
            
            line=str(epoch)+"\t"+str(mse_above_list.mean())+"\t"+str(mse_below_list.mean())+"\t"+str(mse_list.mean())
            print(line)
            logfile.write(line+"\n")
            logfile.flush()
            fsync(logfile.fileno())
            
            
for model_no in range(len(model_bases)):
    model_base = model_bases[model_no]
    name_of_log_file = names_of_log_files[model_no]
    print("Working on model base", model_base)
    make_logfile(name_of_log_file, epochs_of_model, model_base, dataset_tag, threshold_greater_then)
    print("Saved data to", name_of_log_file)
            