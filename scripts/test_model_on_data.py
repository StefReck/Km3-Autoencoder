"Test a model on specific data"
from keras.models import load_model
from util.run_cnn import h5_get_number_of_rows,generate_batches_from_hdf5_file
import argparse
import numpy as np

from get_dataset_info import get_dataset_info
from util.run_cnn import load_zero_center_data

def parse_input():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('models', nargs="+", type=str)

    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
models_to_load = params["models"]
dataset="xzt_broken3"
is_autoencoder=False

dataset_info_dict = get_dataset_info(dataset)
home_path=dataset_info_dict["home_path"]
train_file=dataset_info_dict["train_file"]
test_file=dataset_info_dict["test_file"]
n_bins=dataset_info_dict["n_bins"]
broken_simulations_mode=dataset_info_dict["broken_simulations_mode"]


batchsize=32
filesize_divide=1
train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)/filesize_divide)]]
test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)/filesize_divide)]]

xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=1)


print(test_tuple)

for model_to_load in models_to_load:
    model=load_model(model_to_load)
    print("Testing on model", model_to_load)
    for i, (f, f_size) in enumerate(test_tuple):
        evaluation = model.evaluate_generator(
                generate_batches_from_hdf5_file(f, batchsize=batchsize, n_bins=n_bins, class_type=(2, 'up_down'), is_autoencoder=is_autoencoder, swap_col=None, f_size=f_size, zero_center_image=xs_mean, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict),
                steps=int(f_size / batchsize), max_queue_size=10)

    return_message = 'Test sample results: ' + str(evaluation) + ' (' + str(model.metrics_names) + ')'
    print(return_message)
