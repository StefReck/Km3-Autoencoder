# -*- coding: utf-8 -*-
"""
Calculate the mean activation of the encoded layer of an autoencoder.
The encoded layer is picked out automatically.
Also has the option to plot a histogram of the output of the encoder, and save that
as a pdf multipage.
"""

import numpy as np
import argparse
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import sys
#sys.path.append('../')
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from get_dataset_info import get_dataset_info
#from model_definitions import setup_model
from util.run_cnn import load_zero_center_data, generate_batches_from_hdf5_file, h5_get_number_of_rows

def parse_input():
    parser = argparse.ArgumentParser(description='Get mean activation of encoded layer.')
    parser.add_argument('model_name', nargs="+", type=str)

    args = parser.parse_args()
    params = vars(args)

    return params

#calculate mean and std for 100 batches
calc_mean=True
#make a hist for 1 batch of the flattened encoder output
plot_it=False
#name of pdf, None if no save
to_pdf = "out.pdf"


def get_index_of_encoded_layer(model):
    #Look for last layer that has smallest number of neurons and return its index
    minimum_neurons = np.prod(model.input_shape[1:])
    index_of_encoded_layer=0
    for layer_index,layer in enumerate(model.layers):
        neurons = np.prod(layer.output_shape[1:])
        if neurons <= minimum_neurons:
            minimum_neurons = neurons 
            index_of_encoded_layer=layer_index
    print("Encoded layer:", model.layers[index_of_encoded_layer].name, "with size of bottleneck:", minimum_neurons, "neurons")
    return index_of_encoded_layer

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


def get_output_from_layer(layer_no, model, batch_of_data, train_mode=False):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_no].output])
    layer_output = get_layer_output([batch_of_data,train_mode])[0]
    return layer_output




#train_file=dataset_info_dict["train_file"]
#test_file=dataset_info_dict["test_file"]
#filesize_factor_test=dataset_info_dict["filesize_factor_test"]
#batchsize=dataset_info_dict["batchsize"] #def 32
#filesize_factor=dataset_info_dict["filesize_factor"]
#n_bins=dataset_info_dict["n_bins"]

#train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
#test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]

def get_mean(autoencoder, dataset, no_of_batches):
    index_of_encoded_layer = get_index_of_encoded_layer(autoencoder)
    
    dataset_info_dict=get_dataset_info(dataset)
    generator = setup_generator_testfile(class_type=None, is_autoencoder=True, dataset_info_dict=dataset_info_dict)

    list_of_means_of_events=[]
    for batch_no in range(no_of_batches):       
        batch_of_data = next(generator)[0]
        encoded_output = get_output_from_layer(index_of_encoded_layer, autoencoder, batch_of_data)
        # dimension will be e.g. (32,2,2,3,50)
        # mean(32xmean(2,2,3,50)) = mean(32,2,2,3,50), 
        # ie mean of means is identical to mean of all sample sets (if all sample sets have same sample size)
        #reshape to (32, X)
        encoded_output = np.reshape( encoded_output, ( encoded_output.shape[0], np.prod(encoded_output.shape[1:]) ) )
        means = np.mean(encoded_output, axis=1)
        list_of_means_of_events.extend(means)
    mean_activation = np.mean(list_of_means_of_events)
    standard_deviation = np.std(list_of_means_of_events)
    return mean_activation, standard_deviation


def make_activation_plot(autoencoder, dataset, no_of_batches):
    index_of_encoded_layer = get_index_of_encoded_layer(autoencoder)
    
    dataset_info_dict=get_dataset_info(dataset)
    generator = setup_generator_testfile(class_type=None, is_autoencoder=True, dataset_info_dict=dataset_info_dict)

    list_of_output=np.array([])
    for batch_no in range(no_of_batches):       
        batch_of_data = next(generator)[0]
        encoded_output = get_output_from_layer(index_of_encoded_layer, autoencoder, batch_of_data)
        # dimension will be e.g. (32,2,2,3,50)
        list_of_output=np.append(list_of_output,encoded_output)
        
    fig, ax = plt.subplots(figsize=(8,8))
    ax.hist(list_of_output.flatten(), bins=100)
    ax.set_xlabel("Output of neurons")
    ax.set_ylabel("Number of neurons")
    fig.suptitle("Flattened output of encoder layer")

    return fig, ax


if __name__=="__main__":
    params = parse_input()
    autoencoder_models = params["model_name"]
    
    #autoencoder_model=""
    dataset="xzt"
    no_of_batches = 100
    
    figures_list=[]
    
    for autoencoder_model in autoencoder_models:
        autoencoder = load_model(autoencoder_model)
        if calc_mean == True:
            mean_activation, standard_deviation = get_mean(autoencoder, dataset, no_of_batches)
            print(autoencoder_model, ":", mean_activation, "+-", standard_deviation)
            
        if plot_it == True:
            fig, ax = make_activation_plot(autoencoder, dataset, no_of_batches=1)
            figures_list.append([fig, ax])
            if to_pdf == None:
                plt.show(fig)

if to_pdf != None:
    with PdfPages(to_pdf) as pp:
        for figure in figures_list:
            pp.savefig(figure[0])
            plt.close()
    
        
        
        
        
