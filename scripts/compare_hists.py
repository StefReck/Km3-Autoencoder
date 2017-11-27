# -*- coding: utf-8 -*-
"""
Make 3D Scatter Histogram from 11x13x18 np array, and compare to another one.
Can either plot single histogramms, or two side by side in one plot.
"""
import matplotlib
matplotlib.use('Agg') #dont open plotting windows
from keras.models import load_model
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py

#if MaxUnpooling is present: Load model manually:
from model_definitions import *


def reshape_3d_to_3d(hist_data, filter_small=0):
    #input: 11x13x18
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

def make_3d_plot(grid):
    #input format: [x,y,z,val]
    #maximale count zahl pro bin
    max_value=np.amax(grid)
    min_value=np.amin(grid)
    fraction=(grid[3]-min_value)/max_value

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    #for (xi,yi,zi,val) in grid:
        #drawSphere(xi,yi,zi,r)

    plot = ax.scatter(grid[0],grid[1],grid[2], c=grid[3], s=8*36*fraction)
    cbar=fig.colorbar(plot)
    cbar.set_label('Hits', rotation=270)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def make_3d_plots_xyz(hist_org, hist_pred, suptitle=None):
    #Plot original and predicted histogram side by side in one plot
    #input format: [x,y,z,val]

    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(121, projection='3d', aspect='equal')
    max_value1= np.amax(hist_org)
    min_value1 = np.amin(hist_org) #min value is usually 0, but who knows if the autoencoder screwed up
    fraction1=(hist_org[3]-min_value1)/max_value1
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=8*36*fraction1)
    cbar1=fig.colorbar(plot1,fraction=0.046, pad=0.1)
    cbar1.set_label('Hits', rotation=270, labelpad=0.1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title("Original")
    
    ax2 = fig.add_subplot(122, projection='3d', aspect='equal')
    max_value2=np.amax(hist_pred)
    min_value2=np.amin(hist_pred)
    fraction2=(hist_pred[3]-min_value2)/max_value2
    plot2 = ax2.scatter(hist_pred[0],hist_pred[1],hist_pred[2], c=hist_pred[3], s=8*36*fraction2)
    cbar2=fig.colorbar(plot2,fraction=0.046, pad=0.1)
    cbar2.set_label('Hits', rotation=270, labelpad=0.1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("Prediction")
    
    if suptitle is not None: fig.suptitle(suptitle)

    fig.tight_layout()
    
def make_3d_plots_xzt(hist_org, hist_pred, suptitle=None):
    #Plot original and predicted histogram side by side in one plot
    #input format: [x,z,t,val]
    #maximale count zahl pro bin


    fig = plt.figure(figsize=(8,5))
    
    ax1 = fig.add_subplot(121, projection='3d', aspect='equal')
    max_value1= np.amax(hist_org)
    min_value1 = np.amin(hist_org)
    fraction1=(hist_org[3]-min_value1)/max_value1
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=8*36*fraction1, rasterized=False)
    cbar1=fig.colorbar(plot1,fraction=0.046, pad=0.1)
    cbar1.set_label('Hits', rotation=270, labelpad=0.15)
    ax1.set_xlabel('X')
    ax1.set_xlim([0,11])
    ax1.set_ylabel('Z')
    ax1.set_ylim([0,18])
    ax1.set_zlabel('T')
    ax1.set_zlim([0,50])
    ax1.set_title("Original")
    
    ax2 = fig.add_subplot(122, projection='3d', aspect='equal')
    max_value2=np.amax(hist_pred)
    min_value2=np.amin(hist_pred)
    fraction2=(hist_pred[3]-min_value2)/max_value2
    plot2 = ax2.scatter(hist_pred[0],hist_pred[1],hist_pred[2], c=hist_pred[3], s=8*36*fraction2, rasterized=False)
    cbar2=fig.colorbar(plot2,fraction=0.046, pad=0.1)
    cbar2.set_label('Hits', rotation=270, labelpad=0.15)
    ax2.set_xlabel("X")
    ax2.set_xlim([0,11])
    ax2.set_ylabel('Z')
    ax2.set_ylim([0,18])
    ax2.set_zlabel('T')
    ax2.set_zlim([0,50])

    ax2.set_title("Prediction")
    
    if suptitle is not None: fig.suptitle(suptitle)
    fig.tight_layout()   


def compare_hists_xyz(hist_org, hist_pred, suptitle=None):
    make_3d_plots_xyz(reshape_3d_to_3d(hist_org), reshape_3d_to_3d(hist_pred), suptitle)
    plt.show() 
    
def compare_hists_xzt(hist_org, hist_pred, suptitle=None):
    make_3d_plots_xzt(reshape_3d_to_3d(hist_org), reshape_3d_to_3d(hist_pred), suptitle)
    plt.show() 

def plot_hist(hist):
    make_3d_plot(reshape_3d_to_3d(hist))
    plt.show() 

def save_some_plots_to_pdf( model_file, test_file, zero_center_file, which, plot_file, min_counts=0 ):
    #Only bins with abs. more then min_counts are plotted
    #Open files
    file=h5py.File(test_file , 'r')
    zero_center_image = np.load(zero_center_file)
    
    autoencoder=load_model(model_file)
    #if lambda layers are present:    
    #autoencoder=setup_model("vgg_1_xzt_max", 0)
    #autoencoder.load_weights(model_file, by_name=True)
    #autoencoder.compile(optimizer="adam", loss='mse')

    # event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    labels = file["y"][which]
    hists = file["x"][which]
    
    #Get some hists from the file
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
        pp.attach_note(test_file)
        for i,hist in enumerate(hists):
            suptitle = "Event ID " + str(ids[i]) + "    Energy " + str(energies[i]) + "     Loss: " + str(losses[i])
            make_3d_plots_xzt(reshape_3d_to_3d(hist, min_counts), reshape_3d_to_3d(hists_pred[i], min_counts), suptitle)
            pp.savefig()
            plt.close()
  


if __name__ == '__main__':
    #The Model which is used for predictions
    model_path="models/vgg_3_eps/"
    model_name="trained_vgg_3_eps_autoencoder_epoch20.h5"
    
    #Events to compare
    which = [0,1,2,3,4,5]
    
    #Path to data to predict on (for xzt)
    data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
    test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
    #Path to my Km3_net-Autoencoder folder on HPC:
    home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"
    
    plot_path = home_path + "results/plots/"
    plot_name = model_name[:-3]+"_compare_plot.pdf"
    test_file = data_path + test_data
    zero_center_file= data_path + zero_center_data
    plot_file = plot_path + plot_name
    model_file = home_path + model_path + model_name
    
    
    #Debug:
    """
    model_file="../Models/trained_vgg_0_autoencoder_epoch3.h5"
    test_file = 'Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5'
    zero_center_file = "Daten/train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5_zero_center_mean.npy"
    plot_file = ""+plot_name
    """

    save_some_plots_to_pdf(model_file, test_file, zero_center_file, which, plot_file, min_counts=0.3)





