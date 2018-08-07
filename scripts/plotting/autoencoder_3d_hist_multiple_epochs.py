# -*- coding: utf-8 -*-
"""
Make 3d plots of some events and the autoencoder predictions for one autoencoder model
and multiple epochs.
"""
import matplotlib
matplotlib.use('Agg') #dont open plotting windows

from keras.models import load_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('../')

from get_dataset_info import get_dataset_info
from histogramm_3d_utils import make_plots_from_array


save_name_of_pdf="vgg_3_AE_3d_hists.pdf"
#name of the AE h5 files, up to the epoch number
autoencoder_model_base = "../../models/vgg_3/trained_vgg_3_autoencoder_epoch"
plot_which_epochs = [1,10,60,140,]
number_of_events = 5
#insted of plotting X events, can also explicitly look for a specific event id and plot only that
# None if not
event_id_override = None #737

dataset_tag="xzt"




def get_hists(data_path_of_file, number_of_events):
    #get events from a file
    file=h5py.File(data_path_of_file , 'r')
    #read data, not zero centered
    hists = file["x"][:number_of_events]
    info  = file["y"][:number_of_events]
    return hists, info

def get_specific_id(data_path_of_file, target_event_id):
    #look for the event with a specific id in the file, and return that
    file=h5py.File(data_path_of_file , 'r')
    batchsize = 320
    
    batch_no = 0
    while True:
        info = file["y"][batchsize*batch_no:batchsize*(batch_no+1)]
        event_ids = info[:,0].astype(int)
        where = np.where(event_ids==target_event_id)[0]
        if len(where)!=0:
            where_is_it = batchsize*batch_no + where[0]
            break
            
    hists = file["x"][where_is_it:where_is_it+1]
    info = file["y"][where_is_it:where_is_it+1]
    return hists, info

def predict_on_hists(hists, zero_center_file, autoencoder_model):
    #get predicted image of a batch of hists from autoencoder
    autoencoder = load_model(autoencoder_model)
    zero_center_image = np.load(zero_center_file)
    #zero center and add 1 to the end of the dimensions
    zero_centered_hists = np.subtract( hists.reshape((hists.shape+(1,))), zero_center_image )
    #predict on data
    zero_centered_hists_pred=autoencoder.predict_on_batch(zero_centered_hists)
    #remove zero centering and remove 1 at the end of dimension again,
    #so that the output has the same dimension as the hists input
    hists_pred = np.add(zero_centered_hists_pred, zero_center_image).reshape(hists.shape)
    return hists_pred
    

dataset_info_dict = get_dataset_info(dataset_tag)
data_path_of_file = dataset_info_dict["train_file"]
zero_center_file  = data_path_of_file + "_zero_center_mean.npy"
    
print("Data file:", data_path_of_file)
print("Zero center file:", zero_center_file)

# a list of all the autoencoders for training
autoencoders_list = []
for epoch in plot_which_epochs:
    autoencoders_list.append(autoencoder_model_base + str(epoch) + ".h5")


# read data from file
if event_id_override == None:
    original_image_batch, info = get_hists(data_path_of_file, number_of_events)
else:
    original_image_batch, info = get_specific_id(data_path_of_file, event_id_override)


# make predictions for all the models
predicted_image_batch=[]
for AE_no,autoencoder in enumerate(autoencoders_list):
    #the prediction of all events for a single AE
    print("Predicting for model", autoencoder)
    # dimension e.g. (5,11,13,18) for 5 events
    pred_image_batch = predict_on_hists(original_image_batch, zero_center_file, autoencoder)
    predicted_image_batch.append(pred_image_batch)
#dimension e.g. (3,5,11,13,18) for 3 AEs and 5 events
predicted_image_batch=np.array(predicted_image_batch)


#plot them all in a pdf
# for every event: first the original image, then the predictions from the AEs
print("Event info:\n [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]")
figures = []
for original_image_no,original_image in enumerate(original_image_batch):
    #First the original image
    print("Plotting event no", original_image_no+1)
    event_id = info[original_image_no,0].astype(int)
    print("Event info:", info[original_image_no])
    #[event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]
    org_fig = make_plots_from_array(original_image, suptitle="Original image  Event ID: "+str(event_id))
    figures.append(org_fig)
    plt.close(org_fig)
    
    #then the prediction of all the AEs
    for autoencoder_no, predicted_image in enumerate(predicted_image_batch[:,original_image_no]):
        fig = make_plots_from_array(predicted_image, suptitle="Event ID: "+str(event_id)+"  Autoencoder epoch "+str(plot_which_epochs[autoencoder_no]), min_counts=0.2, titles=["",])
        figures.append(fig)
        plt.close(fig)
    
    
with PdfPages(save_name_of_pdf) as pp:
    for figure in figures:
        pp.savefig(figure)
        plt.close(figure)
        
print("Saved plot to", save_name_of_pdf)
    
