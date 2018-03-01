# -*- coding: utf-8 -*-
"""
Make 3d plots of some events and the autoencoder predictions for one autoencoder model
and multiple epochs.
"""

from keras.models import load_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('../')

from get_dataset_info import get_dataset_info
from histogramm_3d_utils import make_plots_from_array


save_name_of_pdf="test.pdf"
#name of the AE h5 files, up to the epoch number
autoencoder_model_base = "models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_epoch"
plot_which_epochs = [5,6,7,8,9,10]
number_of_events = 2

dataset_tag="xzt"




def get_hists(data_path_of_file, number_of_events, zero_center_file=None, autoencoder_model=None):
    #get events from a file; can also predict on it with an autoencoder
    autoencoder = load_model(autoencoder_model)
    
    file=h5py.File(data_path_of_file , 'r')
    
    #read data, not zero centered
    hists = file["x"][:number_of_events]
    info  = file["y"][:number_of_events]
    
    if autoencoder_model != None:
        zero_center_image = np.load(zero_center_file)
        #zero center and add 1 to the end of the dimensions
        zero_centered_hists = np.subtract( hists.reshape((hists.shape+(1,))), zero_center_image )
        #predict on data
        zero_centered_hists_pred=autoencoder.predict_on_batch(zero_centered_hists)
        #remove zero centering and remove 1 at the end of dimension again
        hists_pred = np.add(zero_centered_hists_pred, zero_center_image).reshape(hists.shape)
        return hists_pred, info
    
    else:
        return hists, info
    
dataset_info_dict = get_dataset_info(dataset_tag)
data_path_of_file = dataset_info_dict["home_path"]+dataset_info_dict["train_file"]
zero_center_file  = data_path_of_file + "_zero_center_mean.npy"
    
print("Data file:", data_path_of_file)
print("Zero center file:", zero_center_file)

figures = []
original_image, info = get_hists(data_path_of_file, number_of_events)
fig = make_plots_from_array(original_image, suptitle="Original image")
figures.append(fig)

for autoencoder_epoch in plot_which_epochs:
    autoencoder_model = autoencoder_model_base + str(autoencoder_epoch) + ".h5"
    pred_image_batch, info  = get_hists(data_path_of_file, number_of_events, 
                                  zero_center_file=zero_center_file, autoencoder_model=autoencoder_model)
    for i,pred_image in enumerate(pred_image_batch):
        event_id = info[i,0].astype(int)
        fig = make_plots_from_array(pred_image, suptitle="Event ID:", event_id, " Autoencoder epoch"+str(autoencoder_epoch))
        figures.append(fig)
    
    
with PdfPages(save_name_of_pdf) as pp:
    for figure in figures:
        pp.savefig(figure)
        plt.close(figure)
        
print("Saved plot to", save_name_of_pdf)
    