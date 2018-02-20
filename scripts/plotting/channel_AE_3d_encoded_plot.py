# -*- coding: utf-8 -*-
"""
Fancy 3d plotting of the encoded layer of a channel Autoencoder 
to show up-down seperation.
"""
import numpy as np
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from get_dataset_info import get_dataset_info
from model_definitions import setup_model
from util.run_cnn import load_zero_center_data, generate_batches_from_hdf5_file, h5_get_number_of_rows
from plotting.make_autoencoder_3d_output_plot import reshape_3d_to_3d


def parse_input():
    parser = argparse.ArgumentParser(description='Make plot of channel AE encoded image')
    parser.add_argument('model_name', type=str)

    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
model_name = params["model_name"]

modeltag="channel_3n"


#which plot to make: 
#encoded: 3d plot of the neurons in the encoded layer
#xyzc: 3d plots of what the vgg gets
plot_type = "xyzc"
#how much of the train file should be gone through for this plot
fraction_of_train_file=1


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

def make_4_plots(plot1, plot2, plot3, plot4, n_bins, titles): 
    binsize_to_name_dict = {11: "X", 13:"Y", 18:"Z", 50:"T", 31:"Channel"}
    
    fig = plt.figure(figsize=(9,9))
    
    for i,plot in enumerate([plot1, plot2, plot3, plot4]):
        ax = fig.add_subplot(221+i, projection='3d')
            
        plot_this = reshape_3d_to_3d(plot, filter_small=0.5)
        plot = ax.scatter(plot_this[0],plot_this[1],plot_this[2], c=plot_this[3], rasterized=True)
      
        cbar=fig.colorbar(plot,fraction=0.046, pad=0.1)
        cbar.ax.set_title('Hits')
        ax.set_xlabel(binsize_to_name_dict[n_bins[0]])
        ax.set_xlim([0,n_bins[0]])
        ax.set_ylabel(binsize_to_name_dict[n_bins[1]])
        ax.set_ylim([0,n_bins[1]])
        ax.set_zlabel(binsize_to_name_dict[n_bins[2]])
        ax.set_zlim([0,n_bins[2]])
        ax.set_title(titles[i])
    fig.tight_layout()
    return fig

if plot_type == "encoded":
    dataset_info_dict=get_dataset_info("xyzc")
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
    filesize_factor=dataset_info_dict["filesize_factor"]
    n_bins=dataset_info_dict["n_bins"]
    
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test*fraction_of_train_file)]]
    
    
    #how many batches will be taken from the file
    steps = int(test_tuple[0][1] / batchsize)
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=dataset_info_dict["n_bins"], n_gpu=1)
    #predictions = encoder.predict_generator(generator, steps=int(test_tuple[0][1] / batchsize), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    encoder = setup_model(modeltag, 1, model_name, additional_options="encoder_only")
    encoder.compile(optimizer="adam", loss='mse')

    generator = setup_generator_testfile( (1, "up_down"), False, dataset_info_dict)
    data, ys = next(generator)
    #data is dimension (batchsize,11,13,18,31)
    #channel AE takes it in that dimension also
    #ys is (32,1)

    is_zero_array = np.where(ys==0)[0]#down
    is_one_array = np.where(ys==1)[0] #up
    print(is_zero_array, is_one_array)
    
    how_many_each=3
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    
    
    for is_zero in is_zero_array[:how_many_each]:
        data_event_0, ys_0 = data[is_zero], ys[is_zero]
        print(ys_0)
        prediction_zero = np.reshape( encoder.predict( np.reshape(data_event_0, (1,)+n_bins) ), n_bins[:-1]+(3,))
        prediction_flat_zero=np.reshape(prediction_zero, (11*13*18, 3))
        prediction_flat_zero=prediction_flat_zero.transpose()
        ax.scatter(prediction_flat_zero[0], prediction_flat_zero[1], prediction_flat_zero[2], c="blue", label="down", rasterized=True)
    
    for is_one in is_one_array[:how_many_each]:
        
        data_event_1, ys_1 = data[is_one], ys[is_one]
        print(ys_1)
        #prediction has dimension (11,13,18,3)
        prediction_one  = np.reshape( encoder.predict( np.reshape(data_event_1, (1,)+n_bins) ), n_bins[:-1]+(3,))
        prediction_flat_one= np.reshape(prediction_one,  (11*13*18, 3))
         
        # make to (x,y,z), shape (3,32)
        prediction_flat_one=prediction_flat_one.transpose()
    
        ax.scatter(prediction_flat_one[0], prediction_flat_one[1], prediction_flat_one[2], c="red", label="up", rasterized=True)
    
    
    plt.legend()
    plt.show()
    
    
    
elif plot_type=="xyzc":
    dataset_info_dict=get_dataset_info("xyzc")
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
    filesize_factor=dataset_info_dict["filesize_factor"]
    n_bins=dataset_info_dict["n_bins"]
    
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test*fraction_of_train_file)]]
    
    
    #how many batches will be taken from the file
    #steps = int(test_tuple[0][1] / batchsize)
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=dataset_info_dict["n_bins"], n_gpu=1)
    #predictions = encoder.predict_generator(generator, steps=int(test_tuple[0][1] / batchsize), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    encoder = setup_model(modeltag, 1, model_name, additional_options="encoder_only")
    encoder.compile(optimizer="adam", loss='mse')
    
    generator = setup_generator_testfile( (1, "up_down"), False, dataset_info_dict, yield_mc_info=True)
    select_id = None
    cycle=True
    while cycle==True:
        data, ys, mc_info = next(generator)
        #data is dimension (batchsize,11,13,18,31)
        #channel AE takes it in that dimension also
        #ys is (32,1)
        #mc info: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
        
        for event_no, event in enumerate(data):
            event_id = mc_info[event_no][0]
            if select_id != None:
                if event_id!=select_id:
                    cycle=False
                    continue
            
            if mc_info[event_no][2]<60:
                continue
            
            print("Mc Info: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]\n", mc_info[event_no])
            #event: (1,11,13,18,31)
            #prediction: (1,11,13,18,3)
            prediction = encoder.predict(event.reshape((1,)+event.shape))
            prediction = np.reshape( prediction, prediction.shape[1:]) #11,13,18,3
            xyz_event = np.add( event, xs_mean )
            xyz_event = np.sum(event.reshape(data.shape[1:]), axis=-1) #11,13,18
            
            titles = ["Summed over channel id", "Neuron 1", "Neuron 2", "Neuron 3"]
            fig = make_4_plots(xyz_event, prediction[:,:,:,0], prediction[:,:,:,1], prediction[:,:,:,2], n_bins[:-1], titles )
            plt.show(fig)
    
    
else:
    print("ploty type unknow")
    raise()
