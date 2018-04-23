# -*- coding: utf-8 -*-
import h5py
import matplotlib.pyplot as plt
import numpy as np

"""
Make a plot that shows what fraction of events from a h5 file are down-going.
"""
datafile = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"



def make_updown_array(datafile):
    file = h5py.File(datafile, "r")
    labels=file["y"]
    energy = labels[:,2]
    dir_z = labels[:,7]
    
    down_going_events=energy[dir_z>0]
    
    average= float(len(down_going_events))/len(energy)
    print("Total percentage of down-going events: ", average)
    
    plot_range=(3,100)
    hist_1d_energy = np.histogram(energy, bins=98, range=plot_range) #häufigkeit von energien
    hist_1d_energy_correct = np.histogram(down_going_events, bins=98, range=plot_range) #häufigkeit von richtigen energien
    
    bin_edges = hist_1d_energy[1]
    hist_1d_energy_accuracy_bins = np.divide(hist_1d_energy_correct[0], hist_1d_energy[0], dtype=np.float32) #rel häufigkeit von richtigen energien
    # For making it work with matplotlib step plot
    #hist_1d_energy_accuracy_bins_leading_zero = np.hstack((0, hist_1d_energy_accuracy_bins))
    bin_edges_centered = bin_edges[:-1] + 0.5
    
    return [bin_edges_centered, hist_1d_energy_accuracy_bins, average]

bin_edges_centered, hist_1d_energy_accuracy_bins, average = make_updown_array(datafile)
#bin_edges_centered, hist_1d_energy_accuracy_bins = np.load("Daten/xzt_test_data_updown.npy")

#The average percentage
average=0.5367258059801829
plt.axhline(average, color="orange", ls="--")
plt_bar_1d_energy_accuracy = plt.step(bin_edges_centered, hist_1d_energy_accuracy_bins, where='mid')

x_ticks_major = np.arange(0, 101, 10)
plt.xticks(x_ticks_major)
plt.yticks(np.arange(0.4,0.6,0.02))
plt.minorticks_on()

plt.xlabel('Energy [GeV]')
plt.ylabel('Fraction')
plt.ylim((0.4, 0.6))
plt.title("Fraction of down-going events in xzt simulated test data")
plt.grid(True)
plt.text(11, average+0.005, "Total avg.: "+str(average*100)[:5]+" %", color="orange", fontsize=10, bbox=dict(facecolor='white', color="white", alpha=0.5))

plt.show()
