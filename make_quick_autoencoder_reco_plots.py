# -*- coding: utf-8 -*-
"""
Make some plots of original event vs autoencoder reco and show them.
"""
import argparse
def parse_input():
    parser = argparse.ArgumentParser(description='Make some plots of original event vs autoencoder reco and show them.')
    parser.add_argument('model', metavar="m", type=str, help='The model that does the predictions.')
    parser.add_argument('dataset_tag', metavar="d", type=str, help='Dataset to use.')
    
    parser.add_argument('how_many', metavar="h", type=int, nargs="?", default=4, help='How many plots of events will be in the pdf')
    parser.add_argument('energy_threshold', metavar="e", type=float, nargs="?", default=0, help='Minimum energy of events for them to be considered')
    parser.add_argument('-z', '--zero_center', action="store_true", help='Use zero-centering?')
    
    return vars(parser.parse_args())
params = parse_input()



from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from histogramm_3d_utils import make_plots_from_array, get_some_hists_from_file
from scripts.get_dataset_info import get_dataset_info
from util.run_cnn import load_zero_center_data, h5_get_number_of_rows
from util.custom_loss_functions import get_custom_objects




model_file = params["model"]
dataset_tag = params["dataset_tag"]
energy_threshold = params["energy_threshold"]
how_many = params["how_many"]
zero_center = params["zero_center"]
       
autoencoder = load_model(model_file, custom_objects=get_custom_objects())

dataset_info_dict = get_dataset_info(dataset_tag)
train_file = dataset_info_dict["train_file"]
test_file=dataset_info_dict["test_file"]
n_bins=dataset_info_dict["n_bins"]
filesize_factor=dataset_info_dict["filesize_factor"]
filesize_factor_test=dataset_info_dict["filesize_factor_test"]
batchsize=dataset_info_dict["batchsize"] #def 32

if zero_center == 1:
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=1)


#array of how_many hists
hists_org = get_some_hists_from_file(train_file, how_many, energy_threshold)
   
#0 center them and add a 1 to shape
if zero_center==1:
    hists_centered = np.subtract(hists_org.reshape((hists_org.shape+(1,))).astype(np.float32), xs_mean)
elif zero_center==0:
    hists_centered = hists_org.reshape((hists_org.shape+(1,))).astype(np.float32)
    
hists_pred = autoencoder.predict_on_batch(hists_centered)

for event_no in range(len(hists_org)):
    fig = make_plots_from_array(hists_org[event_no], hists_pred[event_no])
    plt.show(fig)


