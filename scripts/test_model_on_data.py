"Test a model on specific data"
from keras.models import load_model
from util.run_cnn import h5_get_number_of_rows,generate_batches_from_hdf5_file
import argparse
import numpy as np

def parse_input():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('models', nargs="+", type=str)

    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
models_to_load = params["models"]

data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
filesize_divide=1

test_file=data_path+test_data
zero_center_file = data_path+zero_center_data

batchsize=32
n_bins = (11,18,50,1)
xs_mean = np.load(zero_center_file)

test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)/filesize_divide)]]
print(test_tuple)

for model_to_load in models_to_load:
    model=load_model(model_to_load)
    print("Testing on model", model_to_load)
    for i, (f, f_size) in enumerate(test_tuple):
        evaluation = model.evaluate_generator(
                generate_batches_from_hdf5_file(f, batchsize=batchsize, n_bins=n_bins, class_type=(2, 'up_down'), is_autoencoder=False, swap_col=None, f_size=f_size, zero_center_image=xs_mean),
                steps=int(f_size / batchsize), max_queue_size=10)

    return_message = 'Test sample results: ' + str(evaluation) + ' (' + str(model.metrics_names) + ')'
    print(return_message)
