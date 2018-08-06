# -*- coding: utf-8 -*-
"""
Contains the scripts for plotting 3d histogramms of events and autoencoder outputs.
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py

def reshape_3d_to_3d(hist_data, filter_small=None):
    #input z.B. 11x13x18  (x,y,z)
    #output: [ [x],[y],[z],[val] ]
    #all values with abs<filter_small are removed
    n_bins=hist_data.shape
    """
    tot_bin_no=1
    for i in range(len(n_bins)):
        tot_bin_no=tot_bin_no*n_bins[i]
    """
    #if the input shape is sth like (11,13,18,1), remove the 1 at the end
    if n_bins[-1]==1:
        hist_data = hist_data.reshape(n_bins[:-1])
        print("reshape_3d_to_3d: Changed shape of hist_data from", n_bins, "to", n_bins[:-1])
        n_bins=n_bins[:-1]
    #besser:
    tot_bin_no=np.prod(n_bins)
    
    grid=np.zeros((4,tot_bin_no))
    
    """
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
    """
    
    #better:
    for index, ((x,y,z), value) in enumerate(np.ndenumerate(hist_data)):
        grid[:,index] = (x,y,z,value)
    
    if filter_small is not None:
        bigs = abs(grid[3])>filter_small
        grid=grid[:,bigs] #Bug?
        #grid[3] = np.multiply(grid[3],bigs)
        
    return grid

def size_of_circles(hist, mode="xzt"):
    #Size of the circles in the histogram, depending on the counts of the bin they represent
    #different modes are available, each optimised for different datasets
    max_value = np.amax(hist)
    min_value = np.amin(hist)
    
    if mode=="default":
        size=8*36*((hist[-1]-min_value)/max_value)
        
    elif mode=="xzt":
        size=500*36*((hist[-1]-min_value)/max_value)**2+1
    
    return size


def make_3d_plots(hist_org, n_bins, suptitle, figsize, titles=["Original", "Prediction"], hist_pred=[]):
    #Can plot one or two histogramms (hist_org, hist_pred)
    #n_bins e.g. (11,18,50)
    #input format: e.g. [x,y,z,val]
    
    #for labels on the axes
    binsize_to_name_dict = {11: "X", 13:"Y", 18:"Z", 50:"T", 31:"Channel"}

    fig = plt.figure(figsize=figsize)
    
    if hist_pred != []:
        plot_arrangement = 121
    else:
        plot_arrangement = 111
    
    
    ax1 = fig.add_subplot(plot_arrangement, projection='3d')
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=size_of_circles(hist_org), rasterized=True)
      
    cbar_ticks_org = np.arange(int(hist_org[3].min()),hist_org[3].max()+1,1)
    cbar1=fig.colorbar(plot1,fraction=0.046, pad=0.1, ticks=cbar_ticks_org)
    cbar1.ax.set_title('Hits')
    ax1.set_xlabel(binsize_to_name_dict[n_bins[0]])
    ax1.set_xlim([0,n_bins[0]])
    ax1.set_ylabel(binsize_to_name_dict[n_bins[1]])
    ax1.set_ylim([0,n_bins[1]])
    ax1.set_zlabel(binsize_to_name_dict[n_bins[2]])
    ax1.set_zlim([0,n_bins[2]])
    ax1.set_title(titles[0])
    
    if hist_pred != []:
        ax2 = fig.add_subplot(122, projection='3d')
        plot2 = ax2.scatter(hist_pred[0],hist_pred[1],hist_pred[2], c=hist_pred[3], s=size_of_circles(hist_pred), rasterized=True)
        #Ticks of the color bar: The data can be e.g. from 0.3 to 10, ticks will be at 1,2,...,10
        cbar_ticks = np.arange( np.ceil(hist_pred[3].min()), np.ceil(hist_pred[3].max())+1 ).tolist()

        cbar2=fig.colorbar(plot2,fraction=0.046, pad=0.1, ticks=cbar_ticks)
        cbar2.ax.set_title('Hits')
        ax2.set_xlabel(binsize_to_name_dict[n_bins[0]])
        ax2.set_xlim([0,n_bins[0]])
        ax2.set_ylabel(binsize_to_name_dict[n_bins[1]])
        ax2.set_ylim([0,n_bins[1]])
        ax2.set_zlabel(binsize_to_name_dict[n_bins[2]])
        ax2.set_zlim([0,n_bins[2]])
        ax2.set_title(titles[1])
    
    
    if suptitle is not None: fig.suptitle(suptitle)
    fig.tight_layout() 
        
    return fig

"""
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
"""

def get_title_arrays(y):
    #input: y info from h5 file for a batch
    #Ouput: Multiple arrays with infos of each event for the title as strings
    ids = y[:,0].astype(int)
    particle_type = y[:,1] #Specifies the particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
    energies = y[:,2]
    is_CC = y[:,3]
    dir_z = y[:,7]
    
    #Make an array that contains strings:
    # "Up-going" or "Down-going" 
    #depending on the direction the prt is going
    up_or_down_array=[]
    for dz in dir_z:
        up_or_down = int(np.sign(dz)) # returns -1 if dir_z < 0, 0 if dir_z==0, 1 if dir_z > 0
        direction = "Up-going" if up_or_down>=0 else "Down-going"
        up_or_down_array.append(direction)
        
    #Make an array that contains eg 
    # "Muon-CC" or "Elec-NC"
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
        
    energy_array=[]
    for energy in energies:
        entry = np.round(energy,1)
        energy_array.append("Energy: " + str(entry) + " GeV")
        
    return up_or_down_array, pid_array, energy_array



def make_plots_from_array( array1, array2=[], suptitle=None, min_counts=0.3, titles=["Original", "Prediction"], figsize="auto"):
    #INput: numpy array(s) with shape (a,b,c)
    #output: 3dplot
    n_bins=array1.shape
    
    if figsize=="auto":
        if array2==[]:
            figsize=(6,7)
        else:
            figsize=(12,7)
            
    if array2 != []:
        fig = make_3d_plots(hist_org = reshape_3d_to_3d(array1, min_counts), 
                        hist_pred = reshape_3d_to_3d(array2, min_counts), 
                        n_bins=n_bins, suptitle=suptitle, figsize=figsize,
                        titles=titles)
    else:
        fig = make_3d_plots(hist_org = reshape_3d_to_3d(array1, min_counts), 
                        hist_pred = [], 
                        n_bins=n_bins, suptitle=suptitle, figsize=figsize,
                        titles=titles)
        
    return fig


def get_some_hists_from_file(filepath, how_many, energy_threshold=0):
    #Get some event hists from a dataset
    with h5py.File(filepath, 'r') as file:
        if energy_threshold > 0:
            #Only take events with an energy over a certain threshold
            minimum_energy = energy_threshold #GeV
            only_load_this_many_events=10000 #takes forever otherwise
            print("Applying threshold of", minimum_energy," GeV; only events with higher energy will be considered")
            where=file["y"][:only_load_this_many_events][:, 2]>minimum_energy
            labels=file["y"][:only_load_this_many_events][where,:][:how_many,:]
            hists = file["x"][:only_load_this_many_events][where,:][:how_many,:]
        else:
            #manually select events
            # event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
            print("No energy threshold applied")
            labels = file["y"][:how_many]
            hists = file["x"][:how_many]
    return hists, labels


def get_event_no_from_file(filepath, target_event_id=None, event_track=None):
    #Get an event with a specific event id from a file
    #or alternatively look for an event with the same event track
    hists=[]
    labels=[]
    with h5py.File(filepath, 'r') as file:
        only_load_this_many_events=10000 #takes forever otherwise
        step=0
        while True:
            if event_track==None:
                ids = file["y"][step*only_load_this_many_events:(step+1)*only_load_this_many_events,0]
                if len(ids)==0:
                    print(target_event_id, " was not found.")
                    raise()
                location_locale=np.where(target_event_id==ids)[0]
                
                if len(location_locale)!=0:
                    location = list(step*only_load_this_many_events + location_locale)
                    hists.extend(file["x"][location])
                    labels.extend(file["y"][location])
                    break
                else:
                    step+=1
                
                
            else:
                tracks = file["y"][step*only_load_this_many_events:(step+1)*only_load_this_many_events,1:]
                if len(tracks)==0:
                    print(event_track, " was not found.")
                    raise()
                    
                if event_track[1:] not in tracks:
                    step+=1
                    continue
                else:
                    for location_locale in range(len(tracks)):
                        if event_track[1:]==tracks:
                            location = list(step*only_load_this_many_events + location_locale)
                            hists.extend(file["x"][location])
                            labels.extend(file["y"][location])
                            break
    return hists, labels





def save_some_plots_to_pdf(autoencoder, file, zero_center_file, which, plot_file, min_counts, n_bins, compare_histograms, energy_threshold):
    #Only bins with abs. more then min_counts are plotted
    #Open files
    
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
    
    #Get some hists from the file, and reshape them from eg. (5,11,13,18) to (5,11,13,18,1)
    hists=hists.reshape((hists.shape+(1,))).astype(np.float32)
    
    #0 center them
    zero_center_image = np.load(zero_center_file)
    centered_hists = np.subtract(hists, zero_center_image)
        
    if compare_histograms==True:
        #Predict on 0 centered data with autoencoder
        hists_pred_centered=autoencoder.predict_on_batch(centered_hists).reshape(centered_hists.shape)
        losses=[]
        for i in range(len(centered_hists)):
            losses.append(autoencoder.evaluate(x=centered_hists[i:i+1], y=centered_hists[i:i+1]))
        #Remove centering again, so that empty bins (always 0) dont clutter the view
        hists_pred = np.add(hists_pred_centered, zero_center_image)
        hists_pred=hists_pred.reshape((hists_pred.shape[:-1]))
    
    #change shape from (...,1) to (...)
    hists=hists.reshape((hists.shape[:-1]))
    
    #get info in form of strings for the title
    up_or_down_array, pid_array, energy_array = get_title_arrays(labels)
    
    #make plots and save them in multipage pdf
    with PdfPages(plot_file) as pp:
        print("Saving plot as", plot_file)
        for i,hist in enumerate(hists):
            if compare_histograms == True:
                suptitle = pid_array[i]+"\t\t"+up_or_down_array[i]+"\t\t" + energy_array[i] + "\t\tLoss: " + str(np.round(losses[i],5))
                fig = make_plots_from_array(array1=hist, array2=hists_pred[i], 
                                            min_counts=min_counts, suptitle=suptitle,)

            else:
                suptitle = pid_array[i]+"\t\t"+up_or_down_array[i]+"\t\t" + energy_array[i]
                fig = make_plots_from_array(array1=hist, array2=[], 
                                            min_counts=min_counts, suptitle=suptitle,)
                #fig = make_3d_single_plot(reshape_3d_to_3d(hist, min_counts), n_bins, suptitle, figsizes[1])
            pp.savefig(fig)
            plt.close(fig)
    print("Done.")

