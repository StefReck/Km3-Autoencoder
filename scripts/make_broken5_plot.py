# -*- coding: utf-8 -*-
"""
Look for events with the same event id in the datasets xzt and xzt_broken5 and plot them.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

from get_dataset_info import get_dataset_info
from plotting.histogramm_3d_utils import make_3d_plots, reshape_3d_to_3d, get_title_arrays
from plotting.histogramm_3d_utils import get_event_no_from_file, get_some_hists_from_file, make_plots_from_array

mode="influence"

datainfo_xzt = get_dataset_info("xztc")
train_file_xzt = datainfo_xzt["train_file"]


def make_broken5_manip(hists_temp, chance):
    #Input (X,11,18,50,31) xztc hists
    #Output: (X,11,18,50) manipulated xzt hists
    
    #all doms facing upwards AND having >0 counts have a 30% of getting reduced by one
    
    #1=upwards facing, taken from the paper
    up_mask=np.array([True,]*12 + [False,]*19) 
    #chance for upward facing doms with >0 counts to have one count removed:
    #chance=0.3
        
    #the counts to subtract: upwards facing doms have a 30% chance of getting one count removed
    subt = np.multiply(up_mask, np.random.choice([0,1],size=hists_temp.shape, p=[1-chance,chance]))
    #only remove one count when there is one to begin with
    subt=np.multiply(subt,hists_temp>=1)
    #Ultimately, all doms facing upwards AND having >0 counts have a 30% of getting reduced by one
    hists_temp=hists_temp-subt
    #sum over channel axis to get X,11,18,50 xzt data
    hists_temp=np.sum(hists_temp, axis=-1)
    
    return hists_temp


if mode=="compare_datasets":
    #check if the generate script worked properly by plotting the same event from
    #xztc (summed over c) and xzt_broken5
    
    datainfo_xzt_broken5 = get_dataset_info("xzt_broken5")
    train_file_xzt_broken5 = datainfo_xzt_broken5["train_file"]
    
    hists, labels = get_some_hists_from_file(train_file_xzt, 1, 50)
    
    title_array=get_title_arrays(labels)
    title=title_array[0][0]+title_array[0][1]+title_array[0][2]
    
    hists_b, labels_b = get_event_no_from_file(train_file_xzt_broken5, target_event_id=labels[0][0], event_track=None)
    
    print(labels, labels_b)
    
    #xztc hists haben shape (11,18,50,31)
    hists=np.sum(hists[0], axis=-1)
    
    
    fig = make_plots_from_array(hists, hists_b[0], suptitle=title, min_counts=0, titles=["Original","Manipulation"])
    plt.show(fig)



elif mode=="develop":
    #used for finding a suitable manipulation of the data
    how_many=2
    energy_threshold=50
    chance=0.3
    
    #shape: xztc
    hists, labels = get_some_hists_from_file(train_file_xzt, how_many, energy_threshold)
    title_array=get_title_arrays(labels)
    
    org_hists = np.sum(hists, axis=-1)
    manip_hists = make_broken5_manip(hists, chance)
    
    for i in range(len(hists)):
        title=title_array[0][i]+title_array[1][i]+title_array[2][i]
        fig = make_plots_from_array(org_hists[i], manip_hists[i], suptitle=title, min_counts=0, titles=["Original","Manipulation"])
        plt.show(fig)

elif mode=="influence":
    how_many=10000
    energy_threshold=0
    chance=0.3
    
    #shape: xztc
    hists, labels = get_some_hists_from_file(train_file_xzt, how_many, energy_threshold)
    up_going = labels[:,7]>0

    up_hists=hists[up_going]
    down_hists=hists[np.invert(up_going)]
    
    for channel_id in range(31):
        print("Channel",channel_id, "up and down")
        up_channel = up_hists[:,:,:,:,channel_id]
        down_channel = down_hists[:,:,:,:,channel_id]
        print(np.mean(up_channel)-np.mean(down_channel))
    
    
    
    
    
    
