# -*- coding: utf-8 -*-

#import matplotlib
#matplotlib.use('Agg') #dont open plotting windows


#TODO Debugging!!!


from keras.models import load_model
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import argparse

#if MaxUnpooling is present: Load model manually:
from model_definitions import *

"""
Compare original 3d image of an event with the prediciton of an autoencoder. 
Saves a single plot for multiple files to the same place where the model is loacated at.
"""

def parse_input():
    parser = argparse.ArgumentParser(description='Compare original 3d image of an event with the prediciton of an autoencoder. Saves a single plot for multiple files to the same place where the model is loacated at.')
    parser.add_argument('model', type=str, help='The model taht does the predictions.')
    parser.add_argument('modeltag_lambda_comp', type=str, default=None, nargs="?", help="The modeltag of the model, only if lambda layers are present (which are bugged with load_model).")
  
    args = parser.parse_args()
    params = vars(args)
    return params

params = parse_input()
model_path = params["model"]
lambda_comp_model_tag = params["modeltag_lambda_comp"]


#The Model which is used for predictions
#model_path="/home/woody/capn/mppi013h/Km3-Autoencoder/models/vgg_3_max/trained_vgg_3_max_autoencoder_epoch10.h5"
#if lambda layers are present: Give model tag to load model manually; else None
#lambda_comp_model_tag = None #"vgg_3"


def reshape_3d_to_3d(hist_data, filter_small=0):
    #input z.B. 11x13x18  (x,y,z)
    #output: [ [x],[y],[z],[val] ]
    #all values with abs<filter_small are removed
    n_bins=hist_data.shape
    tot_bin_no=1
    for i in range(len(n_bins)):
        tot_bin_no=tot_bin_no*n_bins[i]
        
    grid=np.zeros((4,tot_bin_no))
    i=0
    for x in range( n_bins[0] ):
        for y in range( n_bins[1] ):
            for z in range( n_bins[2] ):
                val=hist_data[x][y][z]
                grid[0][i]=x
                grid[1][i]=y
                grid[2][i]=z
                grid[3][i]=val
                i=i+1
                
    if filter_small>=0:
        bigs = abs(grid[3])>filter_small
        grid=grid[:,bigs] #Bug?
        #grid[3] = np.multiply(grid[3],bigs)
        
    return grid


def make_3d_plots(hist_org, hist_pred, n_bins, suptitle=None):
    #Plot original and predicted histogram side by side in one plot
    #n_bins e.g. (11,18,50)
    #input format: e.g. [x,y,z,val]

    binsize_to_name_dict = {11: "X", 13:"Y", 18:"Z", 50:"t"}
    
    def size_of_circles(hist):
        max_value = np.amax(hist)
        min_value = np.amin(hist)
        size=8*36*(hist[-1]-min_value)/max_value
        return size

    fig = plt.figure(figsize=(8,5))
    
    ax1 = fig.add_subplot(121, projection='3d', aspect='equal')
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=size_of_circles(hist_org), rasterized=True)
    cbar1=fig.colorbar(plot1,fraction=0.046, pad=0.1)
    cbar1.set_label('Hits', rotation=270, labelpad=0.15)
    ax1.set_xlabel(binsize_to_name_dict[n_bins[0]])
    ax1.set_xlim([0,n_bins[0]])
    ax1.set_ylabel(binsize_to_name_dict[n_bins[1]])
    ax1.set_ylim([0,n_bins[1]])
    ax1.set_zlabel(binsize_to_name_dict[n_bins[2]])
    ax1.set_zlim([0,n_bins[2]])
    ax1.set_title("Original")
    
    ax2 = fig.add_subplot(122, projection='3d', aspect='equal')
    plot2 = ax2.scatter(hist_pred[0],hist_pred[1],hist_pred[2], c=hist_pred[3], s=size_of_circles(hist_pred), rasterized=True)
    cbar2=fig.colorbar(plot2,fraction=0.046, pad=0.1)
    cbar2.set_label('Hits', rotation=270, labelpad=0.15)
    ax2.set_xlabel(binsize_to_name_dict[n_bins[0]])
    ax2.set_xlim([0,n_bins[0]])
    ax2.set_ylabel(binsize_to_name_dict[n_bins[1]])
    ax2.set_ylim([0,n_bins[1]])
    ax2.set_zlabel(binsize_to_name_dict[n_bins[2]])
    ax2.set_zlim([0,n_bins[2]])
    ax2.set_title("Prediction")
    
    if suptitle is not None: fig.suptitle(suptitle)
    fig.tight_layout()   
    
    return fig


def save_some_plots_to_pdf(autoencoder, test_file, zero_center_file, which, plot_file, min_counts=0, n_bins):
    #Only bins with abs. more then min_counts are plotted
    #Open files
    file=h5py.File(test_file , 'r')
    zero_center_image = np.load(zero_center_file)

    # event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    labels = file["y"][which]
    hists = file["x"][which]
    
    #Get some hists from the file, and reshape them from eg. 5,11,13,18 to 5,11,13,18,1
    hists=hists.reshape((hists.shape+(1,))).astype(np.float32)
    #0 center them
    centered_hists = np.subtract(hists, zero_center_image)
    #Predict on 0 centered data
    hists_pred_centered=autoencoder.predict_on_batch(centered_hists).reshape(centered_hists.shape)
    losses=[]
    for i in range(len(centered_hists)):
        losses.append(autoencoder.evaluate(x=centered_hists[i:i+1], y=centered_hists[i:i+1]))
    #Remove centering again, so that empty bins (always 0) dont clutter the view
    hists_pred = np.add(hists_pred_centered, zero_center_image)
    
    #Some infos for the title
    ids = labels[:,0].astype(int)
    energies = labels[:,2].astype(int)

    #test_file is a .h5 file on which the predictions are done
    #center_file is the zero center image
    with PdfPages(plot_file) as pp:
        #pp.attach_note(test_file)
        for i,hist in enumerate(hists):
            suptitle = "Event ID " + str(ids[i]) + "    Energy " + str(energies[i]) + " GeV     Loss: " + str(losses[i])
            make_3d_plots(reshape_3d_to_3d(hist, min_counts), reshape_3d_to_3d(hists_pred[i], min_counts), n_bins, suptitle)
            pp.savefig()
            plt.close()
    



if lambda_comp_model_tag == None:
    autoencoder=load_model(model_file)
else:
    #if lambda layers are present:    
    autoencoder=setup_model(lambda_comp_model_tag, 0)
    autoencoder.load_weights(model_file, by_name=True)
    autoencoder.compile(optimizer="adam", loss='mse')
    

n_bins = autoencoder.input_shape[1:-1] #input_shape ist z.b. None,11,13,18,1

#Events to compare
which = [0,1,2,3,4,5]

#minimum number of counts in a bin for it to be displayed in the histogramms
min_counts=0.3



#Path to data to predict on (for xzt)
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"


plot_file = model_path[:-3]+"_3d_output_plot.pdf"
test_file = data_path + test_data
zero_center_file= data_path + zero_center_data


#Debug:
"""
model_file="../Models/trained_vgg_0_autoencoder_epoch3.h5"
test_file = 'Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5'
zero_center_file = "Daten/train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5_zero_center_mean.npy"
plot_file = ""+plot_name
"""

save_some_plots_to_pdf(autoencoder, test_file, zero_center_file, which, plot_file, min_counts=min_counts, n_bins=n_bins)

    
    