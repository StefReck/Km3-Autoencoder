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


def parse_input():
    parser = argparse.ArgumentParser(description='Make plot of channel AE encoded image')
    parser.add_argument('model_name', type=str)

    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
model_name = params["model_name"]


#how much of the train file should be gone through for this plot
fraction_of_train_file=1


dataset_info_dict=get_dataset_info("xyzc")
encoder = setup_model("channel_3n", 1, model_name, additional_options="encoder_only")
encoder.compile(optimizer="adam", loss='mse')

def setup_generator_testfile(class_type, is_autoencoder, dataset_info_dict):
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
                                    f_size=None, zero_center_image=xs_mean, yield_mc_info=False,
                                    swap_col=None, is_in_test_mode = False)
    return generator


generator = setup_generator_testfile( (1, "up_down"), False, dataset_info_dict)

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

    

data, ys = next(generator)
#data is dimension (batchsize,11,13,18,31)
#channel AE takes it in that dimension also
#ys is (32,1)

is_zero = np.where(ys==0)[0][0]#down
is_one = np.where(ys==1)[0][0] #up

#pick two events that are up/down going
data_event_0, ys_0 = data[is_zero], ys[is_zero]
data_event_1, ys_1 = data[is_one], ys[is_one]

#prediction has dimension (11,13,18,3)
prediction_zero = np.reshape(encoder.predict( np.reshape(data_event_0), (1,)+n_bins ), n_bins)
prediction_one  = np.reshape(encoder.predict( np.reshape(data_event_1), (1,)+n_bins ), n_bins)

prediction_flat_zero=np.reshape(prediction_zero, (11*13*18, 3))
prediction_flat_one= np.reshape(prediction_one,  (11*13*18, 3))
 
# make to (x,y,z), shape (3,32)
prediction_flat_zero=prediction_flat_zero.transpose()
prediction_flat_one=prediction_flat_one.transpose()

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(prediction_flat_zero[0], prediction_flat_zero[1], prediction_flat_zero[2], c="blue", label="down", rasterized=True)
ax.scatter(prediction_flat_one[0], prediction_flat_one[1], prediction_flat_one[2], c="red", label="up", rasterized=True)
plt.legend()
plt.show()

