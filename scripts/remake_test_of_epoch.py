# -*- coding: utf-8 -*-
"""
Evaluate a model on a dataset and print the line to console that is 
usually written into the test log file.
"""
import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset and print the line to console that is usually written into the test log file.')
    parser.add_argument('model', type=str, nargs="+", help='Path to .h5 file of a model.')

    args = parser.parse_args()
    params = vars(args)
    return params["model"]

model_path = parse_input()


from keras.models import load_model
from keras import backend as K

from get_dataset_info import get_dataset_info
from util.run_cnn import evaluate_model, h5_get_number_of_rows, load_zero_center_data



dataset_tag="xzt"
zero_center=True
class_type=[2,"up_down"] #e.g. [2,"up_down"]
is_autoencoder=True

print("\nUsing the following parameters:")
print("Model:", model_path)
print("Dataset:", dataset_tag)
print("Zero centering:", zero_center)
print("Class type:", class_type)
print("Is autoencoder:", is_autoencoder,"\n")




model=load_model(model_path)
dataset_info_dict = get_dataset_info(dataset_tag)

home_path=dataset_info_dict["home_path"]
train_file=dataset_info_dict["train_file"]
test_file=dataset_info_dict["test_file"]
n_bins=dataset_info_dict["n_bins"]
broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
filesize_factor=dataset_info_dict["filesize_factor"]
filesize_factor_test=dataset_info_dict["filesize_factor_test"]
batchsize=dataset_info_dict["batchsize"] #def 32

train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
test_files=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    
n_gpu=(1, 'avolkov')
if zero_center == True:
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=n_gpu[0])
else:
    xs_mean = None
        
    
swap_4d_channels=None
evaluation = evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, 
                            swap_4d_channels, is_autoencoder, broken_simulations_mode, 
                            dataset_info_dict)

metrics = model.metrics_names
print("\n\n")
if "acc" in metrics:
    #loss and accuracy
    print('\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6:.4g}'.format("--epoch--" , str(evaluation[0])[:10], 0, str(evaluation[1])[:10], 0, 0, K.get_value(model.optimizer.lr) ))
else:
    #For autoencoders: only loss
    #history object looks like this: training_hist.history = {'loss': [0.9533379077911377, 0.9494166374206543]} for 2 epochs, this trains only one
    print('\n{0}\t{1}\t{2}\t{3}\t{4:.4g}'.format("--epoch--", str(evaluation)[:10], 0, 0, K.get_value(model.optimizer.lr) ))
print("\n\n")
