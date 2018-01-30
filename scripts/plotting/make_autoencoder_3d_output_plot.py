# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg') #dont open plotting windows

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import argparse

#if MaxUnpooling is present: Load model manually:
#currently defunct, not possible to load lambda models
#from model_definitions import *

"""
Compare original 3d image of an event with the prediciton of an autoencoder. 
Dimensions are recognized automatically.
Saves a single plot for multiple files to the same place where the model is loacated at, e.g.: 
    Original: test/model.h5
    Plot:     test/model_3d_output_plot.pdf
"""


def parse_input():
    parser = argparse.ArgumentParser(description='Compare original 3d image of an event with the prediciton of an autoencoder. Saves a single plot for multiple files to the same place where the model is loacated at.')
    parser.add_argument('model', type=str, help='The model that does the predictions. Saved plot will be the same except with ending _3d_output_plot.h5')
    parser.add_argument('how_many', type=int, help='How many plots of events will be in the pdf')
    parser.add_argument('energy_threshold', type=float, help='Minimum energy of events for them to be considered')
    
    parser.add_argument('only_data_savename', type=str, default="", nargs="?", help='If given, plot of only data without model comparison of xzt data will be saved to given loation.')
    
    #parser.add_argument('modeltag_lambda_comp', type=str, default=None, nargs="?", help="The modeltag of the model, only if lambda layers are present (which are bugged with load_model). CURRENTLY NOT FUNCTIONAL")
  
    args = parser.parse_args()
    params = vars(args)
    return params


debug=False
if debug==False:
    params = parse_input()
    model_file = params["model"]
    how_many = params["how_many"]
    energy_threshold = params["energy_threshold"]
    only_data_savename = params["only_data_savename"]
    
    lambda_comp_model_tag = None #params["modeltag_lambda_comp"]

    #Path to data to predict on (for xzt)
    data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
    test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
    test_file = data_path + test_data
    zero_center_file= data_path + zero_center_data
    
    #Name of output file
    if only_data_savename is not "":
        plot_file = only_data_savename
        compare_histograms=False
    else:
        from keras.models import load_model
        plot_file = model_file[:-3]+"_3d_output_plot.pdf"
        compare_histograms=True
    
else:
    model_file="../Daten/xzt/trained_vgg_3_eps_autoencoder_epoch10.h5"
    test_file = '../Daten/xzt/JTE_KM3Sim_gseagen_elec-CC_3-100GeV-1_1E6-1bin-3_0gspec_ORCA115_9m_2016_100_xzt.h5'
    zero_center_file = "../Daten/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
    plot_file = "test.pdf"
    lambda_comp_model_tag = None
    #Minimum energy for events to be considered
    energy_threshold=50
    which=[0,1,]
    from keras.models import load_model
    compare_histograms=1


#Events to compare, each in a seperate page in the pdf
which = range(0,how_many,1)


#minimum number of counts in a bin for it to be displayed in the histogramms
min_counts=0.3
#Data file to predict on
test_file=h5py.File(test_file , 'r')

if compare_histograms==True:
    #Load autoencoder model
    if lambda_comp_model_tag == None:
        autoencoder=load_model(model_file)
    else:
        #if lambda layers are present:    
        autoencoder=model_definitions.setup_model(lambda_comp_model_tag, 0)
        autoencoder.load_weights(model_file, by_name=True)
        autoencoder.compile(optimizer="adam", loss='mse')
        
    n_bins = autoencoder.input_shape[1:-1] #input_shape ist z.b. None,11,13,18,1
else:
    autoencoder=None
    n_bins = test_file["x"].shape[1:] #shape ist z.B. 4000,11,13,18



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

def size_of_circles(hist):
    #Size of the circles in the histogram, depending on the counts of the bin they represent
    max_value = np.amax(hist)
    min_value = np.amin(hist)
    
    #default
    #size=8*36*((hist[-1]-min_value)/max_value)
    #new: for xzt
    size=500*36*((hist[-1]-min_value)/max_value)**2+1
    
    return size


def make_3d_plots(hist_org, hist_pred, n_bins, suptitle, figsize):
    #Plot original and predicted histogram side by side in one plot
    #n_bins e.g. (11,18,50)
    #input format: e.g. [x,y,z,val]
    

    binsize_to_name_dict = {11: "X", 13:"Y", 18:"Z", 50:"T"}

    fig = plt.figure(figsize=figsize)
    
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=size_of_circles(hist_org), rasterized=True)
      
    cbar1=fig.colorbar(plot1,fraction=0.046, pad=0.1, ticks=np.arange(int(hist_org[3].min()),hist_org[3].max()+1,1))
    cbar1.ax.set_title('Hits')
    ax1.set_xlabel(binsize_to_name_dict[n_bins[0]])
    ax1.set_xlim([0,n_bins[0]])
    ax1.set_ylabel(binsize_to_name_dict[n_bins[1]])
    ax1.set_ylim([0,n_bins[1]])
    ax1.set_zlabel(binsize_to_name_dict[n_bins[2]])
    ax1.set_zlim([0,n_bins[2]])
    ax1.set_title("Original")
    
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot2 = ax2.scatter(hist_pred[0],hist_pred[1],hist_pred[2], c=hist_pred[3], s=size_of_circles(hist_pred), rasterized=True)
    
    cbar2=fig.colorbar(plot2,fraction=0.046, pad=0.1, ticks=[hist_pred[3].min(),]+np.arange(int(hist_pred[3].min()),hist_pred[3].max()+1,1).tolist())
    cbar2.ax.set_title('Hits')
    ax2.set_xlabel(binsize_to_name_dict[n_bins[0]])
    ax2.set_xlim([0,n_bins[0]])
    ax2.set_ylabel(binsize_to_name_dict[n_bins[1]])
    ax2.set_ylim([0,n_bins[1]])
    ax2.set_zlabel(binsize_to_name_dict[n_bins[2]])
    ax2.set_zlim([0,n_bins[2]])
    ax2.set_title("Prediction")
    
    
    fig.suptitle(suptitle)
    fig.tight_layout()   
    
    return fig

def make_3d_single_plot(hist_org, n_bins, title, figsize):
    #Plot original histogram
    #n_bins e.g. (11,18,50)
    #input format: e.g. [x,y,z,val]
    
    binsize_to_name_dict = {11: "X", 13:"Y", 18:"Z", 50:"T"}

    fig = plt.figure(figsize=figsize)
    
    
    ax1 = fig.add_subplot(111, projection='3d')
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=size_of_circles(hist_org), rasterized=True)
      
    cbar1=fig.colorbar(plot1,fraction=0.04, pad=0.1, ticks=np.arange(int(hist_org[3].min()),hist_org[3].max()+1,1))
    cbar1.ax.set_title('Hits')
    ax1.set_xlabel(binsize_to_name_dict[n_bins[0]])
    ax1.set_xlim([0,n_bins[0]])
    ax1.set_ylabel(binsize_to_name_dict[n_bins[1]])
    ax1.set_ylim([0,n_bins[1]])
    ax1.set_zlabel(binsize_to_name_dict[n_bins[2]])
    ax1.set_zlim([0,n_bins[2]])  
    
    ax1.set_title(title)
    fig.tight_layout()   
    
    return fig


def save_some_plots_to_pdf(autoencoder, file, zero_center_file, which, plot_file, min_counts, n_bins, compare_histograms, energy_threshold):
    #Only bins with abs. more then min_counts are plotted
    #Open files
    
    zero_center_image = np.load(zero_center_file)
    
    if energy_threshold > 0:
        #Only take events with an energy over a certain threshold
        minimum_energy = energy_threshold #GeV
        only_load_this_many_events=10000 #takes forever otherwise
        print("Applying threshold of", minimum_energy," GeV; only events with higher energy will be considered")
        where=file["y"][:only_load_this_many_events][:, 2]>minimum_energy
        labels=file["y"][:only_load_this_many_events][where,:][which,:]
        hists = file["x"][:only_load_this_many_events][where,:][which,:]
        
    else:
        #manually select events
        # event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
        print("No energy threshold applied")
        labels = file["y"][which]
        hists = file["x"][which]
    
    #Get some hists from the file, and reshape them from eg. 5,11,13,18 to 5,11,13,18,1
    hists=hists.reshape((hists.shape+(1,))).astype(np.float32)
    #0 center them
    centered_hists = np.subtract(hists, zero_center_image)
    
    if compare_histograms==True:
        #Predict on 0 centered data
        hists_pred_centered=autoencoder.predict_on_batch(centered_hists).reshape(centered_hists.shape)
        losses=[]
        for i in range(len(centered_hists)):
            losses.append(autoencoder.evaluate(x=centered_hists[i:i+1], y=centered_hists[i:i+1]))
        #Remove centering again, so that empty bins (always 0) dont clutter the view
        hists_pred = np.add(hists_pred_centered, zero_center_image)
    
    #Some infos for the title
    ids = labels[:,0].astype(int)
    particle_type = labels[:,1] #Specifies the particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
    energies = np.round(labels[:,2],1)
    is_CC = labels[:,3]
    dir_z = labels[:,7]
    
    #Make an array that contains strings Up or down depending on the direction the prt is going
    up_or_down_array=[]
    for dz in dir_z:
        up_or_down = int(np.sign(dz)) # returns -1 if dir_z < 0, 0 if dir_z==0, 1 if dir_z > 0
        direction = "Up-going" if up_or_down>=0 else "Down-going"
        up_or_down_array.append(direction)
        
    #Make an array that contains eg Muon-CC or Elec-NC
    pid_array=[]
    for i in range(len(particle_type)):
        current_type = "CC" if is_CC[i]==True else "NC"
        if particle_type[i]<0:
            particle_type[i]*=-1
            prefix="Anti-"
        else:
            prefix=""
            
        if particle_type[i]==12:
            part="Elec"
        elif particle_type[i]==14:
            part="Muon"
        elif particle_type[i]==16:
            part="Tau"
        part=prefix+part
        
        pid=part+"-"+ current_type
        pid_array.append(pid)

    #proper layout for xzt:
    figsizes=[(12,7),(6,7)] #[double, single]

    #test_file is a .h5 file on which the predictions are done
    #center_file is the zero center image
    with PdfPages(plot_file) as pp:
        #pp.attach_note(test_file)
        for i,hist in enumerate(hists):
            if compare_histograms == True:
                suptitle = pid_array[i]+"\t\t"+up_or_down_array[i]+"\t\t"+"Energy: " + str(energies[i]) + " GeV\t\tLoss: " + str(np.round(losses[i],5))
                fig = make_3d_plots(reshape_3d_to_3d(hist, min_counts), reshape_3d_to_3d(hists_pred[i], min_counts), n_bins, suptitle, figsizes[0])
            else:
                suptitle = pid_array[i]+"\t\t"+up_or_down_array[i]+"\t\t"+"Energy: " + str(energies[i]) + " GeV"
                fig = make_3d_single_plot(reshape_3d_to_3d(hist, min_counts), n_bins, suptitle, figsizes[1])
            print("Saving plot as", plot_file)
            pp.savefig(fig)
            plt.close(fig)
    


save_some_plots_to_pdf(autoencoder, test_file, zero_center_file, which, plot_file, min_counts=min_counts, n_bins=n_bins, compare_histograms=compare_histograms, energy_threshold=energy_threshold)

    
    