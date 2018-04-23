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

theta = {56.289: [0,4,5,7,8,9],
             72.842: [1,2,3,6,10,11],
             107.158: [12,15,16,23,27,30],
             123.711: [13,17,20,21,28,29],
             148.212: [14,18,19,24,25,26],
             180.0: [22,]}

def make_broken5_manip(hists_temp, sum_channel = True):
    #Input (X,11,18,50,31) xztc hists
    #Output: (X,11,18,50) manipulated xzt hists
    
    #all doms facing upwards AND having >0 counts have a chance% of getting reduced by one
    
    #1=upwards facing, taken from the paper
    up_mask=np.array([True,]*12 + [False,]*19) 
    #chance for upward facing doms with >0 counts to have one count removed:
    #chance
        
    #the counts to subtract: upwards facing doms have a chance% chance of getting one count removed
    #subt = np.random.choice([0,1,2],size=hists_temp.shape, p=[0.3,0.5,0.2])
    subt = np.random.binomial(2, 0.4, size=hists_temp.shape)
    subt = np.multiply(up_mask, subt)
    #subtract
    hists_temp=hists_temp-subt
    #negative counts are not allowed, they are set to 0 instead
    hists_temp = np.clip(hists_temp, 0, None)
    
    #sum over channel axis to get X,11,18,50 xzt data
    if sum_channel == True:
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
    how_many=5
    energy_threshold=0
    chance=0.5
    
    #shape: xztc
    hists, labels = get_some_hists_from_file(train_file_xzt, how_many, energy_threshold)
    title_array=get_title_arrays(labels)
    
    org_hists = np.sum(hists, axis=-1)
    manip_hists = make_broken5_manip(hists, chance)
    
    for i in range(len(hists)):
        title=title_array[0][i]+"   "+title_array[1][i]+"   "+title_array[2][i]
        fig = make_plots_from_array(org_hists[i], manip_hists[i], suptitle=title, min_counts=0, titles=["Original","Manipulation"])
        plt.show(fig)

elif mode=="influence":
    #Show the average count number for every channel for up and down going events.
    how_many=1000
    energy_threshold=0
    chance=0.3
    
    #shape: xztc
    hists, labels = get_some_hists_from_file(train_file_xzt, how_many, energy_threshold)
    
    def make_stats_of_influence(hists, labels):
        up_going = labels[:,7]>0
    
        up_hists=hists[up_going]
        down_hists=hists[np.invert(up_going)]
        
        #counts for every channel, seperate for up and down going events
        average_counts = np.zeros((31,2))
        for channel_id in range(31):
            #counts from up/down going events in a specific channel
            up_channel = up_hists[:,:,:,:,channel_id]
            down_channel = down_hists[:,:,:,:,channel_id]
            
            counts = [np.mean(up_channel), np.mean(down_channel)]
            average_counts[channel_id] = counts
        
        for i,channel_id in enumerate(average_counts):
            print ("Channel",i,"\tup and down:",channel_id[0], channel_id[1])
        
        angle_counts = [[],[],[]]
        for angle in theta:
            #channels with this theta angle
            channels = theta[angle]
            #avg counts of these channels
            this_theta = average_counts[channels]
            this_theta_up = np.mean(this_theta[:,0])
            this_theta_down = np.mean(this_theta[:,1])
            print("Theta", angle,": Up ", this_theta_up, "Down ", this_theta_down)
            angle_counts[0].append(angle)
            angle_counts[1].append(this_theta_up)
            angle_counts[2].append(this_theta_down)
        return angle_counts
    
    angle_counts = make_stats_of_influence(hists,labels)
    manip_hists = make_broken5_manip(hists, chance, sum_channel=False)
    manip_angle_counts=make_stats_of_influence(manip_hists,labels)
    
    plt.plot(angle_counts[0], angle_counts[1], "o", label="Up")
    plt.plot(angle_counts[0], angle_counts[2], "o", label="Down")
    
    plt.plot(manip_angle_counts[0], manip_angle_counts[1], "o", label="Up manip")
    plt.plot(manip_angle_counts[0], manip_angle_counts[2], "o", label="Down manip")
    
    plt.xlabel("Polar angle")
    plt.ylabel("Mean counts")
    plt.legend()
    plt.grid()
    plt.show()
    
    """
    theta_counts_array=[[],[],[]]
    for angel in theta:
        for channel in theta[angle]:
            theta_counts_array[0].append(angel)
            theta_counts_array[1].append(average_counts[channel][0])
            theta_counts_array[2].append(average_counts[channel][1])
            
    plt.plot(theta_counts_array[0], theta_counts_array[1], "o-", label="Up")
    plt.plot(theta_counts_array[0], theta_counts_array[2], "o-", label="Down")
    plt.legend()
    plt.grid()
    plt.show()
    """
    
    
