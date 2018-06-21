# -*- coding: utf-8 -*-
"""
Make some plots of original event vs autoencoder reco and show them.
"""
import argparse
def parse_input():
    parser = argparse.ArgumentParser(description='Make some plots of original event vs autoencoder reco and show them.')
    parser.add_argument('model', metavar="m", type=str, help='The model that does the predictions.')
    parser.add_argument('dataset_tag', metavar="d", type=str, help='Dataset to use.')
    
    parser.add_argument('how_many', metavar="h", type=int, nargs="?", default=4, help='How many plots of events will be generated.')
    parser.add_argument('energy_threshold', metavar="e", type=float, nargs="?", default=0, help='Minimum energy of events for them to be considered')
    parser.add_argument('-z', '--no_zero_center', action="store_true", help='Use zero-centering?')
    
    return vars(parser.parse_args())
params = parse_input()



from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from plotting.histogramm_3d_utils import make_plots_from_array, get_some_hists_from_file
from get_dataset_info import get_dataset_info
from util.run_cnn import load_zero_center_data, h5_get_number_of_rows
from util.custom_loss_functions import get_custom_objects
from model_definitions import setup_model



model_file = params["model"]
dataset_tag = params["dataset_tag"]
energy_threshold = params["energy_threshold"]
how_many = params["how_many"]
no_zero_center = params["no_zero_center"]

autoencoder = load_model(model_file, custom_objects=get_custom_objects())

if "vgg_6_200_advers" in model_file:
    #This is an adversary network
    #Only leave the autoencoder part over
    only_autoencoder = setup_model("vgg_5_200",0)
    for i,layer in enumerate(only_autoencoder.layers):
        layer.set_weights(autoencoder.layers[i].get_weights())
    gan_model = autoencoder
    autoencoder = only_autoencoder
else:
    gan_model = None

dataset_info_dict = get_dataset_info(dataset_tag)
train_file = dataset_info_dict["train_file"]
test_file=dataset_info_dict["test_file"]
n_bins=dataset_info_dict["n_bins"]
filesize_factor=dataset_info_dict["filesize_factor"]
filesize_factor_test=dataset_info_dict["filesize_factor_test"]
batchsize=dataset_info_dict["batchsize"] #def 32


#array of how_many hists
hists_org, labels = get_some_hists_from_file(train_file, how_many, energy_threshold)
   
#0 center them and add a 1 to shape
if no_zero_center==False:
    print("Zero centering active")
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=1)
    hists_centered = np.subtract(hists_org.reshape((hists_org.shape+(1,))).astype(np.float32), xs_mean)
    
    hists_pred = np.add(autoencoder.predict_on_batch(hists_centered), xs_mean)
    print(gan_model.predict_on_batch(hists_centered))
    
    hists_pred=hists_pred.reshape((hists_pred.shape[:-1]))
elif no_zero_center==True:
    print("No zero centering")
    hists_centered = hists_org.reshape((hists_org.shape+(1,))).astype(np.float32)
    
    hists_pred = autoencoder.predict_on_batch(hists_centered)
    print(gan_model.predict_on_batch(hists_centered))
    
    hists_pred=hists_pred.reshape((hists_pred.shape[:-1]))


for event_no in range(len(hists_org)):
    fig = make_plots_from_array(array1=hists_org[event_no], array2=hists_pred[event_no])
    plt.show(fig)


